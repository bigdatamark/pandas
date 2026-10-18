"""
Provide a generic structure to support window functions,
similar to how we have a Groupby object.
"""

# [MODULE DOC]
# Core implementation of fixed/variable window (rolling) computations and the
# shared machinery reused by expanding and exponentially weighted windows.
# Public contents (classes): the base window class (construction, validation,
# column/table application of kernels, iteration over windows, selection
# support, aggregation); its groupby variant (computes windows within each
# group and rebuilds a hierarchical result index); the weighted-window class
# (window types such as triangular or gaussian, computed with the compiled
# window kernels); a mixin implementing the statistics shared by rolling and
# expanding windows (count, apply, pipe, sum, max, min, mean, median, std,
# var, skew, sem, kurt, first, last, quantile, rank, nunique, cov, corr); the
# rolling class (fixed, offset-based or custom-indexer windows); and the
# rolling groupby class.
# Relationships: window bounds come from the indexer classes in the core
# indexers objects module; numerical work is done by compiled kernels in the
# window aggregations extension and optional numba kernels (window numba
# module) through the executor; pairwise statistics use the flexible binary
# moment helper in the window common module; expanding and ewm modules
# subclass or reuse these classes.
# Import-time side effects: none beyond imports.

from __future__ import annotations

import copy
from datetime import timedelta
from functools import partial
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    Self,
    cast,
    final,
    overload,
)

import numpy as np

from pandas._libs.tslibs import (
    BaseOffset,
    Timedelta,
    to_offset,
)
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import set_module

from pandas.core.dtypes.common import (
    ensure_float64,
    is_bool,
    is_integer,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import notna

from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import (
    ResamplerWindowApply,
    reconstruct_func,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
    BaseIndexer,
    FixedWindowIndexer,
    GroupbyIndexer,
    VariableWindowIndexer,
)
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    TimedeltaIndex,
)
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
    get_jit_arguments,
    maybe_use_numba,
    prepare_function_arguments,
)
from pandas.core.window.common import (
    flex_binary_moment,
    zsqrt,
)
from pandas.core.window.numba_ import (
    generate_manual_numpy_nan_agg_with_axis,
    generate_numba_apply_func,
    generate_numba_table_func,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import (
        Hashable,
        Iterator,
        Sized,
    )

    from pandas._typing import (
        ArrayLike,
        NDFrameT,
        QuantileInterpolation,
        P,
        TimeUnit,
        T,
        WindowingRankType,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.ops import BaseGrouper

from pandas.core.arrays.datetimelike import dtype_to_unit


class BaseWindow(SelectionMixin):
    # [CLASS DOC]
    # Base class for all windowing objects (rolling, weighted window,
    # expanding, exponentially weighted). Inherits column selection support
    # from the selection mixin.
    # State: obj (series or data frame being windowed), on (column label or
    # index naming the axis windows are computed along, or none for the
    # object's index), closed (which window edges are inclusive), step,
    # window (size, offset, or custom indexer), min_periods, center, win_type,
    # method (single or table), the resolved on-axis index, the cached window
    # size in integer nanoseconds for offset windows (none otherwise) and the
    # current column selection. The class-level attribute list names the
    # constructor arguments copied when selecting sub-objects and printed by
    # the representation; the exclusions set is empty.
    # Key methods: validation, the apply family that turns a kernel into a
    # result (series, column-wise, table-wise, pairwise, numba), aggregate,
    # iteration over windows, attribute-style column access.
    # Invariant: validation runs at the end of construction, so an existing
    # object always has consistent arguments.
    """Provides utilities for performing windowing operations."""

    _attributes: list[str] = []
    exclusions: frozenset[Hashable] = frozenset()
    _on: Index

    def __init__(
        self,
        obj: NDFrame,
        window=None,
        min_periods: int | None = None,
        center: bool | None = False,
        win_type: str | None = None,
        on: str | Index | None = None,
        closed: str | None = None,
        step: int | None = None,
        method: str = "single",
        *,
        selection=None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores all arguments as attributes, resolves the on-axis and then
        # validates. Inputs: obj (series or data frame); window (integer,
        # time offset, or indexer; default none); min_periods (default none,
        # resolved later); center (default false); win_type (weighted window
        # name, default none); on (column label of the frame, an index object,
        # or none for the object's own index); closed (right, both, left,
        # neither, or none); step (default none); method (single or table,
        # default single); selection (keyword-only column selection, default
        # none).
        # The on-axis becomes the object's index, the given index, or the
        # named column wrapped as an index. Raises a value error if on is
        # something else (message: must be a column of a frame, an index or
        # none). Then runs validation, which may raise further errors.
        self.obj = obj
        self.on = on
        self.closed = closed
        self.step = step
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.win_type = win_type
        self.method = method
        self._win_freq_i8: int | None = None
        if self.on is None:
            self._on = self.obj.index
        elif isinstance(self.on, Index):
            self._on = self.on
        elif isinstance(self.obj, ABCDataFrame) and self.on in self.obj.columns:
            self._on = Index(self.obj[self.on])
        else:
            raise ValueError(
                f"invalid on specified as {self.on}, "
                "must be a column (of DataFrame), an Index or None"
            )

        self._selection = selection
        self._validate()

    def _validate(self) -> None:
        # [FUNCTION DOC]
        # HELPER: argument validation hook (subclasses extend it). Returns
        # nothing; raises a value error when: center is neither boolean nor
        # none; min_periods is not an integer, is negative, or exceeds an
        # integer window; closed is not one of right, both, left, neither;
        # method is not table or single; step is not an integer or is
        # negative. Raises a type error if obj is neither series nor frame.
        # For a custom indexer window, checks that its bounds method has
        # exactly the same parameter names as the base indexer's, else a value
        # error.
        # KNOWN ISSUE: the error message for an invalid method lacks the
        # closing quote after the word single.
        if self.center is not None and not is_bool(self.center):
            raise ValueError("center must be a boolean")
        if self.min_periods is not None:
            if not is_integer(self.min_periods):
                raise ValueError("min_periods must be an integer")
            if self.min_periods < 0:
                raise ValueError("min_periods must be >= 0")
            if is_integer(self.window) and self.min_periods > self.window:
                raise ValueError(
                    f"min_periods {self.min_periods} must be <= window {self.window}"
                )
        if self.closed is not None and self.closed not in [
            "right",
            "both",
            "left",
            "neither",
        ]:
            raise ValueError("closed must be 'right', 'left', 'both' or 'neither'")
        if not isinstance(self.obj, (ABCSeries, ABCDataFrame)):
            raise TypeError(f"invalid type: {type(self)}")
        if isinstance(self.window, BaseIndexer):
            # Validate that the passed BaseIndexer subclass has
            # a get_window_bounds with the correct signature.
            get_window_bounds_signature = inspect.signature(
                self.window.get_window_bounds
            ).parameters.keys()
            expected_signature = inspect.signature(
                BaseIndexer().get_window_bounds
            ).parameters.keys()
            if get_window_bounds_signature != expected_signature:
                raise ValueError(
                    f"{type(self.window).__name__} does not implement "
                    f"the correct signature for get_window_bounds"
                )
        if self.method not in ["table", "single"]:
            raise ValueError("method must be 'table' or 'single")
        if self.step is not None:
            if not is_integer(self.step):
                raise ValueError("step must be an integer")
            if self.step < 0:
                raise ValueError("step must be >= 0")

    def _check_window_bounds(
        self, start: np.ndarray, end: np.ndarray, num_vals: int
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: sanity check of window bounds. start and end are the bound
        # arrays, num_vals the number of values windowed. Raises a value error
        # if the arrays differ in length, or if their length is not the number
        # of values divided by the step (step treated as one when unset or
        # zero), rounded up. Returns nothing.
        if len(start) != len(end):
            raise ValueError(
                f"start ({len(start)}) and end ({len(end)}) bounds must be the "
                f"same length"
            )
        if len(start) != (num_vals + (self.step or 1) - 1) // (self.step or 1):
            raise ValueError(
                f"start and end bounds ({len(start)}) must be the same length "
                f"as the object ({num_vals}) divided by the step ({self.step}) "
                f"if given and rounded up"
            )

    def _slice_axis_for_step(self, index: Index, result: Sized | None = None) -> Index:
        # [FUNCTION DOC]
        # HELPER: returns the index entries matching a stepped result. If
        # result is none or has the same length as the index, the index is
        # returned unchanged; otherwise every step-th label starting at the
        # first.
        """
        Slices the index for a given result and the preset step.
        """
        return (
            index
            if result is None or len(result) == len(index)
            else index[:: self.step]
        )

    def _validate_numeric_only(self, name: str, numeric_only: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: rejects the numeric-only option for one-dimensional
        # non-numeric data. name is the operation name used in the message;
        # numeric_only the user flag. Raises a not-implemented error if the
        # selected object is one-dimensional, the flag is true and its dtype
        # is not numeric. Returns nothing otherwise.
        """
        Validate numeric_only argument, raising if invalid for the input.

        Parameters
        ----------
        name : str
            Name of the operator (kernel).
        numeric_only : bool
            Value passed by user.
        """
        if (
            self._selected_obj.ndim == 1
            and numeric_only
            and not is_numeric_dtype(self._selected_obj.dtype)
        ):
            raise NotImplementedError(
                f"{type(self).__name__}.{name} does not implement numeric_only"
            )

    def _make_numeric_only(self, obj: NDFrameT) -> NDFrameT:
        # [FUNCTION DOC]
        # HELPER: returns the frame restricted to numeric columns, excluding
        # time-delta columns (booleans follow the numeric selection rules).
        """Subset DataFrame to numeric columns.

        Parameters
        ----------
        obj : DataFrame

        Returns
        -------
        obj subset to numeric-only columns.
        """
        result = obj.select_dtypes(include=["number"], exclude=["timedelta"])
        return result

    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        # [FUNCTION DOC]
        # HELPER: prepares the data to window. If a column was named as the
        # on-axis (and the object is a frame), that column is removed from the
        # result by reindexing the columns; if numeric_only (default false) and
        # the object is a frame, keeps only numeric columns. Returns the
        # prepared object (the input is not modified). The docstring below is
        # outdated: no block splitting happens here.
        """
        Split data into blocks & return conformed data.
        """
        # filter out the on from the object
        if self.on is not None and not isinstance(self.on, Index) and obj.ndim == 2:
            obj = obj.reindex(columns=obj.columns.difference([self.on], sort=False))
        if obj.ndim > 1 and numeric_only:
            obj = self._make_numeric_only(obj)
        return obj

    def _gotitem(self, key, ndim, subset=None):
        # [FUNCTION DOC]
        # HELPER: selection hook used by column indexing. key is the column
        # label or list; ndim the requested dimensionality (unused here);
        # subset (default the whole object) is the object to select from.
        # Returns a new window object of the same class built from the subset,
        # copying the constructor attributes named in the attribute list, with
        # the inferred selection set. The original is not mutated.
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # create a new object to prevent aliasing
        if subset is None:
            subset = self.obj

        # we need to make a shallow copy of ourselves
        # with the same groupby
        kwargs = {attr: getattr(self, attr) for attr in self._attributes}

        selection = self._infer_selection(key, subset)
        new_win = type(self)(subset, selection=selection, **kwargs)
        return new_win

    def __getattr__(self, attr: str):
        # [FUNCTION DOC]
        # Fallback attribute lookup. Internal names are looked up normally;
        # if attr is a column label of the wrapped object, returns that column
        # selection (as item access would). Otherwise raises an attribute
        # error naming the class and attribute.
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def _dir_additions(self):
        # [FUNCTION DOC]
        # HELPER: names added to interactive attribute completion; delegates
        # to the wrapped object (its column labels).
        return self.obj._dir_additions()

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the class name followed by comma-separated name=value pairs
        # in square brackets, for each constructor attribute in the attribute
        # list that is set (not none) and whose name does not start with an
        # underscore.
        """
        Provide a nice str repr of our rolling object.
        """
        attrs_list = (
            f"{attr_name}={getattr(self, attr_name)}"
            for attr_name in self._attributes
            if getattr(self, attr_name, None) is not None and attr_name[0] != "_"
        )
        attrs = ",".join(attrs_list)
        return f"{type(self).__name__} [{attrs}]"

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Generator yielding, for each window, the slice of the (selected,
        # on-axis indexed, prepared) object that falls into it. Window bounds
        # come from the window indexer using the configured min periods,
        # centering, closedness and step; they are validated, and slices are
        # yielded by position pairing starts with ends. Raises a value error
        # from bound validation if the indexer is inconsistent.
        obj = self._selected_obj.set_axis(self._on)
        obj = self._create_data(obj)
        indexer = self._get_window_indexer()

        start, end = indexer.get_window_bounds(
            num_values=len(obj),
            min_periods=self.min_periods,
            center=self.center,
            closed=self.closed,
            step=self.step,
        )
        self._check_window_bounds(start, end, len(obj))

        for s, e in zip(start, end, strict=True):
            result = obj.iloc[slice(s, e)]
            yield result

    def _prep_values(self, values: ArrayLike) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: converts an array (numpy or extension) to a 64-bit float
        # numpy array for the compiled kernels, replacing infinite values
        # with missing. Raises a not-implemented error for datetime-like
        # dtypes needing integer conversion, and a type error if conversion
        # to float fails. Extension arrays are converted with missing mapped
        # to NaN. Returns the new array (the input is not modified).
        """Convert input to numpy arrays for Cython routines"""
        if needs_i8_conversion(values.dtype):
            raise NotImplementedError(
                f"ops for {type(self).__name__} for this "
                f"dtype {values.dtype} are not implemented"
            )
        # GH #12373 : rolling functions error on float32 data
        # make sure the data is coerced to float64
        try:
            if isinstance(values, ExtensionArray):
                values = values.to_numpy(np.float64, na_value=np.nan)
            else:
                values = ensure_float64(values)
        except (ValueError, TypeError) as err:
            raise TypeError(f"cannot handle this type -> {values.dtype}") from err

        # Convert inf to nan for C funcs
        inf = np.isinf(values)
        if inf.any():
            values = np.where(inf, np.nan, values)

        return values

    def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None:
        # [FUNCTION DOC]
        # HELPER: puts the on-axis column back into a result frame (mutating
        # it in place; returns nothing). Does nothing when no on option was
        # given or the on-axis equals the object's index. Otherwise builds a
        # series of the on values labelled like the original column and: if
        # the result already has a column with that name, overwrites it; if
        # the name is an index level name of the result, does nothing; if the
        # column existed in the selected object, inserts it at the position
        # that keeps its relative order among surviving columns; otherwise
        # appends it at the end.
        # if we have an 'on' column we want to put it back into
        # the results in the same location
        from pandas import Series

        if self.on is not None and not self._on.equals(obj.index):
            name = self._on.name
            extra_col = Series(self._on, index=self.obj.index, name=name, copy=False)
            if name in result.columns:
                # TODO: sure we want to overwrite results?
                result[name] = extra_col
            elif name in result.index.names:
                pass
            elif name in self._selected_obj.columns:
                # insert in the same location as we had in _selected_obj
                old_cols = self._selected_obj.columns
                new_cols = result.columns
                old_loc = old_cols.get_loc(name)
                overlap = new_cols.intersection(old_cols[:old_loc])
                new_loc = len(overlap)
                result.insert(new_loc, name, extra_col)
            else:
                # insert at the end
                result[name] = extra_col

    @property
    def _index_array(self) -> npt.NDArray[np.int64] | None:
        # [FUNCTION DOC]
        # HELPER property: the on-axis as an array of 64-bit integers when it
        # is a period, datetime or time-delta index (underlying integer
        # representation), or an arrow-backed date/time-delta typed index
        # (converted to integers); none for any other axis type.
        # TODO: why do we get here with e.g. MultiIndex?
        if isinstance(self._on, (PeriodIndex, DatetimeIndex, TimedeltaIndex)):
            return self._on.asi8
        elif isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM":
            return self._on.to_numpy(dtype=np.int64)
        return None

    def _resolve_output(self, out: DataFrame, obj: DataFrame) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: finalises a frame result. out is the computed frame and obj
        # the input frame. Raises a data error (no numeric types to
        # aggregate) if the result has no columns while the input had some. If
        # both are empty of columns, returns the input cast to 64-bit float.
        # Otherwise restores the on-axis column and returns out.
        """Validate and finalize result."""
        if out.shape[1] == 0 and obj.shape[1] > 0:
            raise DataError("No numeric types to aggregate")
        if out.shape[1] == 0:
            return obj.astype("float64")

        self._insert_on_column(out, obj)
        return out

    def _get_window_indexer(self) -> BaseIndexer:
        # [FUNCTION DOC]
        # HELPER: chooses the object that computes window bounds. A custom
        # indexer given as the window is returned as is; if the window is a
        # time offset (cached integer nanosecond size set), returns a variable
        # indexer over the on-axis integers with that size and the centering
        # flag; otherwise a fixed-size indexer using the window size.
        """
        Return an indexer class that will compute the window start and end bounds
        """
        if isinstance(self.window, BaseIndexer):
            return self.window
        if self._win_freq_i8 is not None:
            return VariableWindowIndexer(
                index_array=self._index_array,
                window_size=self._win_freq_i8,
                center=self.center,
            )
        return FixedWindowIndexer(window_size=self.window)

    def _apply_series(
        self, homogeneous_func: Callable[..., ArrayLike], name: str | None = None
    ) -> Series:
        # [FUNCTION DOC]
        # HELPER: applies a per-array kernel wrapper to a one-dimensional
        # selection. homogeneous_func maps a float array to the result array;
        # name is the operation name (default none). For the count operation,
        # the data is first replaced by integer not-missing indicators so
        # date-like types are supported. Raises a data error (no numeric types
        # to aggregate) if values cannot be prepared. Returns a series of the
        # result with the index sliced for the step and the original name.
        """
        Series version of _apply_columnwise
        """
        obj = self._create_data(self._selected_obj)

        if name == "count":
            # GH 12541: Special case for count where we support date-like types
            obj = notna(obj).astype(int)
        try:
            values = self._prep_values(obj._values)
        except (TypeError, NotImplementedError) as err:
            raise DataError("No numeric types to aggregate") from err

        result = homogeneous_func(values)
        index = self._slice_axis_for_step(obj.index, result)
        return obj._constructor(result, index=index, name=obj.name)

    def _apply_columnwise(
        self,
        homogeneous_func: Callable[..., ArrayLike],
        name: str,
        numeric_only: bool = False,
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: applies the kernel wrapper column by column. Inputs:
        # homogeneous_func (maps a float array to the result array), name
        # (operation name; count gets special not-missing handling) and
        # numeric_only (default false). Validates the numeric-only flag first
        # (not-implemented error for non-numeric series). A one-dimensional
        # selection is delegated to the series version. For frames: prepares
        # the data, converts each column to float (raising a data error naming
        # the dtype if impossible), applies the function, and assembles a frame
        # with the stepped index and the same column labels, then finalises it
        # (which raises a data error if nothing numeric remained).
        """
        Apply the given function to the DataFrame broken down into homogeneous
        sub-frames.
        """
        self._validate_numeric_only(name, numeric_only)
        if self._selected_obj.ndim == 1:
            return self._apply_series(homogeneous_func, name)

        obj = self._create_data(self._selected_obj, numeric_only)
        if name == "count":
            # GH 12541: Special case for count where we support date-like types
            obj = notna(obj).astype(int)
            obj._mgr = obj._mgr.consolidate()

        taker = []
        res_values = []
        for i, arr in enumerate(obj._iter_column_arrays()):
            # GH#42736 operate column-wise instead of block-wise
            # As of 2.0, hfunc will raise for nuisance columns
            try:
                arr = self._prep_values(arr)
            except (TypeError, NotImplementedError) as err:
                raise DataError(
                    f"Cannot aggregate non-numeric type: {arr.dtype}"
                ) from err
            res = homogeneous_func(arr)
            res_values.append(res)
            taker.append(i)

        index = self._slice_axis_for_step(
            obj.index, res_values[0] if len(res_values) > 0 else None
        )
        df = type(obj)._from_arrays(
            res_values,
            index=index,
            columns=obj.columns.take(taker),
            verify_integrity=False,
        )

        return self._resolve_output(df, obj)

    def _apply_tablewise(
        self,
        homogeneous_func: Callable[..., ArrayLike],
        name: str | None = None,
        numeric_only: bool = False,
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: applies the kernel wrapper to the whole frame as one 2D float
        # array (method table). Inputs: homogeneous_func, name (default none,
        # unused) and numeric_only (default false). Raises a value error for
        # one-dimensional input. Builds the result frame with the stepped
        # index and, if the result width differs from the input's, stepped
        # columns; then finalises it.
        """
        Apply the given function to the DataFrame across the entire object
        """
        if self._selected_obj.ndim == 1:
            raise ValueError("method='table' not applicable for Series objects.")
        obj = self._create_data(self._selected_obj, numeric_only)
        values = self._prep_values(obj.to_numpy())
        result = homogeneous_func(values)
        index = self._slice_axis_for_step(obj.index, result)
        columns = (
            obj.columns
            if result.shape[1] == len(obj.columns)
            else obj.columns[:: self.step]
        )
        out = obj._constructor(result, index=index, columns=columns)

        return self._resolve_output(out, obj)

    def _apply_pairwise(
        self,
        target: DataFrame | Series,
        other: DataFrame | Series | None,
        pairwise: bool | None,
        func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series],
        numeric_only: bool,
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: applies a two-object statistic. target is the primary
        # object; other the second object or none (use target itself, and
        # default pairwise to true); pairwise the pairing mode; func the
        # per-pair function; numeric_only restricts frames to numeric
        # columns. Raises a value error if other is neither a series nor a
        # frame. Delegates the combination to the flexible binary moment
        # helper (pairwise coerced to a boolean) and returns its result.
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
        target = self._create_data(target, numeric_only)
        if other is None:
            other = target
            # only default unset
            pairwise = True if pairwise is None else pairwise
        elif not isinstance(other, (ABCDataFrame, ABCSeries)):
            raise ValueError("other must be a DataFrame or Series")
        elif other.ndim == 2 and numeric_only:
            other = self._make_numeric_only(other)

        return flex_binary_moment(target, other, func, pairwise=bool(pairwise))

    def _apply(
        self,
        func: Callable[..., Any],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: core driver for kernel based statistics. func is a compiled
        # kernel taking the values, window start and end arrays, the minimum
        # periods and any extra positional arguments; name is the operation
        # name; numeric_only (default false) restricts frames; numba_args
        # (default empty) are extra positional arguments appended for the
        # kernel; extra keyword arguments are accepted but not used.
        # Behavior: obtains the window indexer, and sets min periods to the
        # given value or else the indexer's window size. Defines a wrapper that
        # returns an empty copy for empty input, otherwise computes the bounds
        # (with the configured centering, closedness, step), validates them,
        # and calls the kernel with floating point warnings suppressed.
        # Dispatch: method single goes column by column, otherwise the whole
        # table at once. Returns an object like the input (series or frame).
        # Errors come from the dispatch targets (data error, value error).
        """
        Rolling statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
        name : str,
        numba_args : tuple
            args to be passed when func is a numba func
        **kwargs
            additional arguments for rolling function and window function

        Returns
        -------
        y : type of input
        """
        window_indexer = self._get_window_indexer()
        min_periods = (
            self.min_periods
            if self.min_periods is not None
            else window_indexer.window_size
        )

        def homogeneous_func(values: np.ndarray):
            # [FUNCTION DOC]
            # HELPER: nested wrapper handed to the dispatch methods. Takes a
            # float array; returns a copy if it is empty, else the kernel
            # result for it (see the nested calculation helper), computed with
            # numpy floating point error reporting disabled.
            # calculation function

            if values.size == 0:
                return values.copy()

            def calc(x):
                # [FUNCTION DOC]
                # HELPER: computes window bounds for the array x (length,
                # min periods, centering, closedness, step), validates them,
                # and returns the kernel applied to x with the bounds, the
                # minimum periods and the extra numba arguments.
                start, end = window_indexer.get_window_bounds(
                    num_values=len(x),
                    min_periods=min_periods,
                    center=self.center,
                    closed=self.closed,
                    step=self.step,
                )
                self._check_window_bounds(start, end, len(x))

                return func(x, start, end, min_periods, *numba_args)

            with np.errstate(all="ignore"):
                result = calc(values)

            return result

        if self.method == "single":
            return self._apply_columnwise(homogeneous_func, name, numeric_only)
        else:
            return self._apply_tablewise(homogeneous_func, name, numeric_only)

    def _numba_apply(
        self,
        func: Callable[..., Any],
        engine_kwargs: dict[str, bool] | None = None,
        **func_kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: runs a numba kernel through the shared executor. func is the
        # per-window kernel; engine_kwargs (default none) holds numba jit
        # flags; extra keyword arguments are forwarded to the compiled
        # aggregator. Min periods defaults to the indexer's window size.
        # Prepares the data as a float array (reshaped to a column if
        # one-dimensional), computes and validates window bounds, builds a
        # shared aggregator with all dtypes mapped to float (to match the
        # cython implementation, even though that is not strictly right) and
        # applies it to the transposed values. The result is transposed back,
        # indexed with the stepped index (and stepped columns for frames).
        # Returns a series (squeezed, named like the input) for one-dimensional
        # data, otherwise a finalised frame (data error if no numeric columns).
        window_indexer = self._get_window_indexer()
        min_periods = (
            self.min_periods
            if self.min_periods is not None
            else window_indexer.window_size
        )
        obj = self._create_data(self._selected_obj)
        values = self._prep_values(obj.to_numpy())
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        start, end = window_indexer.get_window_bounds(
            num_values=len(values),
            min_periods=min_periods,
            center=self.center,
            closed=self.closed,
            step=self.step,
        )
        self._check_window_bounds(start, end, len(values))
        # For now, map everything to float to match the Cython impl
        # even though it is wrong
        # TODO: Could preserve correct dtypes in future
        # xref #53214
        dtype_mapping = executor.float_dtype_mapping
        aggregator = executor.generate_shared_aggregator(
            func,
            dtype_mapping,
            is_grouped_kernel=False,
            **get_jit_arguments(engine_kwargs),
        )
        result = aggregator(
            values.T, start=start, end=end, min_periods=min_periods, **func_kwargs
        ).T
        index = self._slice_axis_for_step(obj.index, result)
        if obj.ndim == 1:
            result = result.squeeze()
            out = obj._constructor(result, index=index, name=obj.name)
            return out
        else:
            columns = self._slice_axis_for_step(obj.columns, result.T)
            out = obj._constructor(result, index=index, columns=columns)
            return self._resolve_output(out, obj)

    def aggregate(self, func=None, *args, **kwargs):
        # [FUNCTION DOC]
        # Aggregates with one or more functions. func is a function, name,
        # list or mapping (default none); extra arguments and keyword
        # arguments are passed along. Detects named-aggregation relabeling
        # from the keyword arguments. Runs the generic window aggregation
        # machinery; if the result is a frame and relabeling was requested,
        # reorders columns and applies the new labels. If the machinery
        # produced nothing (none), falls back to applying func as a custom
        # function in non-raw mode. Returns the result. Also available under
        # the shorter alias name.
        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if isinstance(result, ABCDataFrame) and relabeling:
            result = result.iloc[:, order]
            result.columns = columns
        if result is None:
            return self.apply(func, raw=False, args=args, kwargs=kwargs)
        return result

    agg = aggregate


class BaseWindowGroupby(BaseWindow):
    # [CLASS DOC]
    # Mixin-style base for windowing within groups of a groupby object.
    # Subclass of the base window class. Extra state: _grouper (the grouper
    # providing group names, codes, levels and row positions per group) and
    # _as_index (whether group keys stay in the result index). The attribute
    # list adds the grouper. Overrides: construction (drops grouping columns,
    # rejects step), applying kernels (rebuilds a hierarchical result index of
    # group labels plus original index), pairwise application, data
    # preparation (reorders rows so groups are contiguous) and selection.
    # Invariant: computations run over data ordered group by group.
    """
    Provide the groupby windowing facilities.
    """

    _grouper: BaseGrouper
    _as_index: bool
    _attributes: list[str] = ["_grouper"]

    def __init__(
        self,
        obj: DataFrame | Series,
        *args,
        _grouper: BaseGrouper,
        _as_index: bool = True,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the groupby window object. obj is the series or frame; extra
        # positional and keyword arguments go to the base initializer;
        # _grouper (required keyword) must be a grouper object, else a value
        # error; _as_index (default true) controls keeping group keys in the
        # result index. Removes the grouping columns from obj (ignoring ones
        # that are absent), and raises a not-implemented error if a step was
        # given.
        from pandas.core.groupby.ops import BaseGrouper

        if not isinstance(_grouper, BaseGrouper):
            raise ValueError("Must pass a BaseGrouper object.")
        self._grouper = _grouper
        self._as_index = _as_index
        # GH 32262: It's convention to keep the grouping column in
        # groupby.<agg_func>, but unexpected to users in
        # groupby.rolling.<agg_func>
        obj = obj.drop(columns=self._grouper.names, errors="ignore")
        # GH 15354
        if kwargs.get("step") is not None:
            raise NotImplementedError("step not implemented for groupby")
        super().__init__(obj, *args, **kwargs)

    def _apply(
        self,
        func: Callable[..., Any],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: same arguments as the base driver (kernel, operation name,
        # numeric-only flag, extra numba arguments, extra keywords). Runs the
        # base computation on group-ordered data, then rebuilds the result
        # index as a multi-index whose first levels are the group key levels
        # (codes reordered to follow group order) and whose remaining levels
        # are the original index levels. Group key columns that were kept as
        # index levels of the object are dropped from the result columns when
        # only part of the keys are index names. If _as_index is false, the
        # group levels are moved back into columns. Returns the result.
        result = super()._apply(
            func,
            name,
            numeric_only,
            numba_args,
            **kwargs,
        )
        # Reconstruct the resulting MultiIndex
        # 1st set of levels = group by labels
        # 2nd set of levels = original DataFrame/Series index
        grouped_object_index = self.obj.index
        grouped_index_name = [*grouped_object_index.names]
        groupby_keys = copy.copy(self._grouper.names)
        result_index_names = groupby_keys + grouped_index_name

        drop_columns = [
            key
            for key in self._grouper.names
            if key not in self.obj.index.names or key is None
        ]

        if len(drop_columns) != len(groupby_keys):
            # Our result will have still kept the column in the result
            result = result.drop(columns=drop_columns, errors="ignore")

        codes = self._grouper.codes
        levels = copy.copy(self._grouper.levels)

        group_indices = self._grouper.indices.values()
        if group_indices:
            indexer = np.concatenate(list(group_indices))
        else:
            indexer = np.array([], dtype=np.intp)
        codes = [c.take(indexer) for c in codes]

        # if the index of the original dataframe needs to be preserved, append
        # this index (but reordered) to the codes/levels from the groupby
        if grouped_object_index is not None:
            idx = grouped_object_index.take(indexer)
            if not isinstance(idx, MultiIndex):
                idx = MultiIndex.from_arrays([idx])
            codes.extend(list(idx.codes))
            levels.extend(list(idx.levels))

        result_index = MultiIndex(
            levels, codes, names=result_index_names, verify_integrity=False
        )

        result.index = result_index
        if not self._as_index:
            result = result.reset_index(level=list(range(len(groupby_keys))))
        return result

    def _apply_pairwise(
        self,
        target: DataFrame | Series,
        other: DataFrame | Series | None,
        pairwise: bool | None,
        func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series],
        numeric_only: bool,
    ) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: pairwise statistic within groups; same arguments as the
        # base pairwise method. Drops grouping columns from the target, runs
        # the base computation, then prepends the group levels to the result's
        # index. If other is given and its length differs from some group's
        # length (the second object is not group-aligned), each group's result
        # rows are reindexed against the full result and concatenated, with
        # group labels repeated per original result row and factorised into
        # codes and levels. Otherwise the group codes are reordered by group
        # row positions and repeated per target column. The result's own index
        # (flat or multi-level) supplies the trailing levels; names are group
        # names followed by the result's index names. Returns the result with
        # the combined multi-index.
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
        # Manually drop the grouping column first
        target = target.drop(columns=self._grouper.names, errors="ignore")
        result = super()._apply_pairwise(target, other, pairwise, func, numeric_only)
        # 1) Determine the levels + codes of the groupby levels
        if other is not None and not all(
            len(group) == len(other) for group in self._grouper.indices.values()
        ):
            # GH 42915
            # len(other) != len(any group), so must reindex (expand) the result
            # from flex_binary_moment to a "transform"-like result
            # per groupby combination
            old_result_len = len(result)
            result = concat(
                [
                    result.take(gb_indices).reindex(result.index)
                    for gb_indices in self._grouper.indices.values()
                ]
            )

            gb_pairs = (
                com.maybe_make_list(pair) for pair in self._grouper.indices.keys()
            )
            groupby_codes = []
            groupby_levels = []
            # e.g. [[1, 2], [4, 5]] as [[1, 4], [2, 5]]
            for gb_level_pair in map(list, zip(*gb_pairs, strict=True)):
                labels = np.repeat(np.array(gb_level_pair), old_result_len)
                codes, levels = factorize(labels)
                groupby_codes.append(codes)
                groupby_levels.append(levels)
        else:
            # pairwise=True or len(other) == len(each group), so repeat
            # the groupby labels by the number of columns in the original object
            groupby_codes = self._grouper.codes
            # error: Incompatible types in assignment (expression has type
            # "List[Index]", variable has type "List[Union[ndarray, Index]]")
            groupby_levels = self._grouper.levels  # type: ignore[assignment]

            group_indices = self._grouper.indices.values()
            if group_indices:
                indexer = np.concatenate(list(group_indices))
            else:
                indexer = np.array([], dtype=np.intp)

            if target.ndim == 1:
                repeat_by = 1
            else:
                repeat_by = len(target.columns)
            groupby_codes = [
                np.repeat(c.take(indexer), repeat_by) for c in groupby_codes
            ]
        # 2) Determine the levels + codes of the result from super()._apply_pairwise
        if isinstance(result.index, MultiIndex):
            result_codes = list(result.index.codes)
            result_levels = list(result.index.levels)
            result_names = list(result.index.names)
        else:
            idx_codes, idx_levels = factorize(result.index)
            result_codes = [idx_codes]
            result_levels = [idx_levels]
            result_names = [result.index.name]

            # 3) Create the resulting index by combining 1) + 2)
        result_codes = groupby_codes + result_codes
        result_levels = groupby_levels + result_levels
        result_names = self._grouper.names + result_names

        result_index = MultiIndex(
            result_levels, result_codes, names=result_names, verify_integrity=False
        )
        result.index = result_index
        return result

    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        # [FUNCTION DOC]
        # HELPER: for non-empty data, reorders rows so each group's rows are
        # contiguous (concatenated group positions), then defers to the base
        # preparation with the numeric-only flag (default false). Returns the
        # prepared object.
        """
        Split data into blocks & return conformed data.
        """
        # Ensure the object we're rolling over is monotonically sorted relative
        # to the groups
        # GH 36197
        if not obj.empty:
            groupby_order = np.concatenate(list(self._grouper.indices.values())).astype(
                np.int64
            )
            obj = obj.take(groupby_order)
        return super()._create_data(obj, numeric_only)

    def _gotitem(self, key, ndim, subset=None):
        # [FUNCTION DOC]
        # HELPER: selection hook. If an on option is set, replaces the subset
        # with the object re-indexed by the on-axis so the axis survives
        # selection; then defers to the base hook with key, ndim and that
        # subset and returns the new groupby window object.
        # we are setting the index on the actual object
        # here so our index is carried through to the selected obj
        # when we do the splitting for the groupby
        if self.on is not None:
            # GH 43355
            subset = self.obj.set_index(self._on)
        return super()._gotitem(key, ndim, subset=subset)


@set_module("pandas.api.typing")
class Window(BaseWindow):
    # [CLASS DOC]
    # Rolling window with non-uniform weights from a named scipy window
    # function (win_type). Subclass of the base window class; no new state
    # beyond the cached weight generator function (looked up by name from the
    # scipy window module at validation time). Supported statistics: sum,
    # mean, var, std, plus aggregate. Restrictions enforced in validation:
    # win_type must be a valid scipy window name, window must be a
    # non-negative integer (no custom indexers, no time offsets), and only the
    # single-column method is supported. Extra keyword arguments of the
    # statistics methods are forwarded to the scipy window generator.
    """
    Provide rolling window calculations.

    Parameters
    ----------
    window : int, timedelta, str, offset, or BaseIndexer subclass
        Interval of the moving window.

        If an integer, the delta between the start and end of each window.
        The number of points in the window depends on the ``closed`` argument.

        If a timedelta, str, or offset, the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes.
        To learn more about the offsets & frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        If a BaseIndexer subclass, the window boundaries
        based on the defined ``get_window_bounds`` method. Additional rolling
        keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
        ``step`` will be passed to ``get_window_bounds``.

    min_periods : int, default None
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

        For a window that is specified by an offset, ``min_periods`` will default to 1.

        For a window that is specified by an integer, ``min_periods`` will default
        to the size of the window.

    center : bool, default False
        If False, set the window labels as the right edge of the window index.

        If True, set the window labels as the center of the window index.

    win_type : str, default None
        If ``None``, all points are evenly weighted.

        If a string, it must be a valid `scipy.signal window function
        <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.

        Certain Scipy window types require additional parameters to be passed
        in the aggregation function. The additional parameters must match
        the keywords specified in the Scipy window type method signature.

    on : str, optional
        For a DataFrame, a column label or Index level on which
        to calculate the rolling window, rather than the DataFrame's index.

        Provided integer column is ignored and excluded from result since
        an integer index is not used to calculate the rolling window.

    closed : str, default None
        Determines the inclusivity of points in the window

        If ``'right'``, uses the window (first, last] meaning the last point
        is included in the calculations.

        If ``'left'``, uses the window [first, last) meaning the first point
        is included in the calculations.

        If ``'both'``, uses the window [first, last] meaning all points in
        the window are included in the calculations.

        If ``'neither'``, uses the window (first, last) meaning the first
        and last points in the window are excluded from calculations.

        () and [] are referencing open and closed set
        notation respetively.

        Default ``None`` (``'right'``).

    step : int, default None
        Evaluate the window at every ``step`` result, equivalent to slicing as
        ``[::step]``. ``window`` must be an integer. Using a step argument other
        than None or 1 will produce a result with a different shape than the input.

    method : str {'single', 'table'}, default 'single'

        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).

        This argument is only implemented when specifying ``engine='numba'``
        in the method call.

    Returns
    -------
    pandas.api.typing.Window or pandas.api.typing.Rolling
        An instance of Window is returned if ``win_type`` is passed. Otherwise,
        an instance of Rolling is returned.

    See Also
    --------
    expanding : Provides expanding transformations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    See :ref:`Windowing Operations <window.generic>` for further usage details
    and examples.

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

    **window**

    Rolling sum with a window length of 2 observations.

    >>> df.rolling(2).sum()
         B
    0  NaN
    1  1.0
    2  3.0
    3  NaN
    4  NaN

    Rolling sum with a window span of 2 seconds.

    >>> df_time = pd.DataFrame(
    ...     {"B": [0, 1, 2, np.nan, 4]},
    ...     index=[
    ...         pd.Timestamp("20130101 09:00:00"),
    ...         pd.Timestamp("20130101 09:00:02"),
    ...         pd.Timestamp("20130101 09:00:03"),
    ...         pd.Timestamp("20130101 09:00:05"),
    ...         pd.Timestamp("20130101 09:00:06"),
    ...     ],
    ... )

    >>> df_time
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  2.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    >>> df_time.rolling("2s").sum()
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  3.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    Rolling sum with forward looking windows with 2 observations.

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0

    **min_periods**

    Rolling sum with a window length of 2 observations, but only needs a minimum of 1
    observation to calculate a value.

    >>> df.rolling(2, min_periods=1).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  2.0
    4  4.0

    **center**

    Rolling sum with the result assigned to the center of the window index.

    >>> df.rolling(3, min_periods=1, center=True).sum()
         B
    0  1.0
    1  3.0
    2  3.0
    3  6.0
    4  4.0

    >>> df.rolling(3, min_periods=1, center=False).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  3.0
    4  6.0

    **step**

    Rolling sum with a window length of 2 observations, minimum of 1 observation to
    calculate a value, and a step of 2.

    >>> df.rolling(2, min_periods=1, step=2).sum()
         B
    0  0.0
    2  3.0
    4  4.0

    **win_type**

    Rolling sum with a window length of 2, using the Scipy ``'gaussian'``
    window type. ``std`` is required in the aggregation function.

    >>> df.rolling(2, win_type="gaussian").sum(std=3)
              B
    0        NaN
    1   0.986207
    2   2.958621
    3        NaN
    4        NaN

    **on**

    Rolling sum with a window length of 2 days.

    >>> df = pd.DataFrame(
    ...     {
    ...         "A": [
    ...             pd.to_datetime("2020-01-01"),
    ...             pd.to_datetime("2020-01-01"),
    ...             pd.to_datetime("2020-01-02"),
    ...         ],
    ...         "B": [1, 2, 3],
    ...     },
    ...     index=pd.date_range("2020", periods=3),
    ... )

    >>> df
                        A  B
    2020-01-01 2020-01-01  1
    2020-01-02 2020-01-01  2
    2020-01-03 2020-01-02  3

    >>> df.rolling("2D", on="A").sum()
                        A    B
    2020-01-01 2020-01-01  1.0
    2020-01-02 2020-01-01  3.0
    2020-01-03 2020-01-02  6.0
    """

    _attributes = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "on",
        "closed",
        "step",
        "method",
    ]

    def _validate(self) -> None:
        # [FUNCTION DOC]
        # HELPER: extends base validation. Raises a value error if win_type is
        # not a string or is not a window function name in the optional scipy
        # window module (import error with a hint if scipy is missing); stores
        # the found weight generator. Raises a not-implemented error for custom
        # indexer windows and for any method other than single; raises a value
        # error if the window is not an integer of zero or more.
        super()._validate()

        if not isinstance(self.win_type, str):
            raise ValueError(f"Invalid win_type {self.win_type}")
        signal = import_optional_dependency(
            "scipy.signal.windows", extra="Scipy is required to generate window weight."
        )
        self._scipy_weight_generator = getattr(signal, self.win_type, None)
        if self._scipy_weight_generator is None:
            raise ValueError(f"Invalid win_type {self.win_type}")

        if isinstance(self.window, BaseIndexer):
            raise NotImplementedError(
                "BaseIndexer subclasses not implemented with win_types."
            )
        if not is_integer(self.window) or self.window < 0:
            raise ValueError("window must be an integer 0 or greater")

        if self.method != "single":
            raise NotImplementedError("'single' is the only supported method type.")

    def _center_window(self, result: np.ndarray, offset: int) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: shifts a weighted result so labels sit at window centres.
        # result is the computed array; offset the number of leading entries
        # to drop. When offset is positive returns a copy without the first
        # offset entries; otherwise returns the input unchanged.
        """
        Center the result in the window for weighted rolling aggregations.
        """
        if offset > 0:
            lead_indexer = [slice(offset, None)]
            result = np.copy(result[tuple(lead_indexer)])
        return result

    def _apply(
        self,
        func: Callable[[np.ndarray, int, int], np.ndarray],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: weighted-window driver. func is a compiled weighted kernel
        # taking values, the weight array and min periods; name is the
        # operation name; numeric_only (default false) restricts frames;
        # numba_args is accepted but unused; extra keyword arguments are the
        # parameters of the scipy window (for example a standard deviation).
        # Builds the weights by calling the scipy generator with the window
        # size and the keyword arguments. The centering offset is half the
        # weights length (rounded down, after subtracting one) when centering,
        # else zero. For each column, appends offset missing values, runs the
        # kernel (with errors suppressed; min periods defaults to the number
        # of weights), converts the memoryview result to an array and, if
        # centering, drops the leading offset entries. Applies column-wise and
        # finally takes every step-th row of the result. Returns series/frame.
        """
        Rolling with weights statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
        name : str,
        numeric_only : bool, default False
            Whether to only operate on bool, int, and float columns
        numba_args : tuple
            unused
        **kwargs
            additional arguments for scipy windows if necessary

        Returns
        -------
        y : type of input
        """
        # "None" not callable  [misc]
        window = self._scipy_weight_generator(  # type: ignore[misc]
            self.window, **kwargs
        )
        offset = (len(window) - 1) // 2 if self.center else 0

        def homogeneous_func(values: np.ndarray):
            # [FUNCTION DOC]
            # HELPER: nested per-array wrapper: empty input gives a copy;
            # otherwise runs the kernel with floating point error reporting
            # off, converts the result to an array, and trims the centering
            # offset if centering is on. Returns the array.
            # calculation function

            if values.size == 0:
                return values.copy()

            def calc(x):
                # [FUNCTION DOC]
                # HELPER: pads the array with offset missing values at the end
                # and returns the weighted kernel applied with the weights and
                # the effective minimum periods.
                additional_nans = np.full(offset, np.nan)
                x = np.concatenate((x, additional_nans))
                return func(
                    x,
                    window,
                    self.min_periods if self.min_periods is not None else len(window),
                )

            with np.errstate(all="ignore"):
                # Our weighted aggregations return memoryviews
                result = np.asarray(calc(values))

            if self.center:
                result = self._center_window(result, offset)

            return result

        return self._apply_columnwise(homogeneous_func, name, numeric_only)[
            :: self.step
        ]

    def aggregate(self, func=None, *args, **kwargs):
        # [FUNCTION DOC]
        # Aggregates the weighted window with one or more functions. func is a
        # function, name, list or mapping (default none); extra arguments and
        # keyword arguments are forwarded. Uses the generic window aggregation
        # machinery; if it yields nothing (func is a plain callable), calls
        # func directly with this window object and returns that. Also
        # available under the shorter alias name. Unlike the base class, no
        # relabeling support and no apply fallback.
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series/DataFrame or
            when passed to Series/DataFrame.apply.

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
        DataFrame.aggregate : Similar DataFrame method.
        Series.aggregate : Similar Series method.

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
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
        A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2, win_type="boxcar").agg("mean")
            A    B    C
        0  NaN  NaN  NaN
        1  1.5  4.5  7.5
        2  2.5  5.5  8.5
        """
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            # these must apply directly
            result = func(self)

        return result

    agg = aggregate

    def sum(self, numeric_only: bool = False, **kwargs):
        # [FUNCTION DOC]
        # Weighted rolling sum. numeric_only (default false) restricts frames
        # to numeric columns; keyword arguments configure the scipy window
        # (for example its standard deviation). Runs the weighted-sum kernel
        # through the weighted driver and returns an object of the input type.
        """
        Calculate the rolling weighted window sum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.sum : Aggregating sum for Series.
        DataFrame.sum : Aggregating sum for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type="gaussian"))
        <class 'pandas.api.typing.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method
        (`sum` in this case):

        >>> ser.rolling(2, win_type="gaussian").sum(std=3)
        0         NaN
        1    0.986207
        2    5.917243
        3    6.903450
        4    9.862071
        dtype: float64
        """
        window_func = window_aggregations.roll_weighted_sum
        # error: Argument 1 to "_apply" of "Window" has incompatible type
        # "Callable[[ndarray, ndarray, int], ndarray]"; expected
        # "Callable[[ndarray, int, int], ndarray]"
        return self._apply(
            window_func,  # type: ignore[arg-type]
            name="sum",
            numeric_only=numeric_only,
            **kwargs,
        )

    def mean(self, numeric_only: bool = False, **kwargs):
        # [FUNCTION DOC]
        # Weighted rolling mean; same parameters as the weighted sum, using the
        # weighted-mean kernel.
        """
        Calculate the rolling weighted window mean.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.mean : Aggregating mean for Series.
        DataFrame.mean : Aggregating mean for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type="gaussian"))
        <class 'pandas.api.typing.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type="gaussian").mean(std=3)
        0    NaN
        1    0.5
        2    3.0
        3    3.5
        4    5.0
        dtype: float64
        """
        window_func = window_aggregations.roll_weighted_mean
        # error: Argument 1 to "_apply" of "Window" has incompatible type
        # "Callable[[ndarray, ndarray, int], ndarray]"; expected
        # "Callable[[ndarray, int, int], ndarray]"
        return self._apply(
            window_func,  # type: ignore[arg-type]
            name="mean",
            numeric_only=numeric_only,
            **kwargs,
        )

    def var(self, ddof: int = 1, numeric_only: bool = False, **kwargs):
        # [FUNCTION DOC]
        # Weighted rolling variance. ddof (default one) is the delta degrees
        # of freedom bound into the weighted-variance kernel; numeric_only and
        # scipy window keyword arguments as for the weighted sum. Any name
        # keyword passed in (used internally by the standard deviation) is
        # discarded before delegating.
        """
        Calculate the rolling weighted window variance.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.var : Aggregating var for Series.
        DataFrame.var : Aggregating var for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type="gaussian"))
        <class 'pandas.api.typing.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type="gaussian").var(std=3)
        0     NaN
        1     0.5
        2     8.0
        3     4.5
        4    18.0
        dtype: float64
        """
        window_func = partial(window_aggregations.roll_weighted_var, ddof=ddof)
        kwargs.pop("name", None)
        return self._apply(window_func, name="var", numeric_only=numeric_only, **kwargs)

    def std(self, ddof: int = 1, numeric_only: bool = False, **kwargs):
        # [FUNCTION DOC]
        # Weighted rolling standard deviation: computes the weighted variance
        # with the same arguments (labelling the operation as the standard
        # deviation) and returns its element-wise square root, with negative
        # values (rounding noise) clamped to zero by the safe square root
        # helper.
        """
        Calculate the rolling weighted window standard deviation.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Keyword arguments to configure the ``SciPy`` weighted window type.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.std : Aggregating std for Series.
        DataFrame.std : Aggregating std for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type="gaussian"))
        <class 'pandas.api.typing.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type="gaussian").std(std=3)
        0         NaN
        1    0.707107
        2    2.828427
        3    2.121320
        4    4.242641
        dtype: float64
        """
        return zsqrt(
            self.var(ddof=ddof, name="std", numeric_only=numeric_only, **kwargs)
        )


class RollingAndExpandingMixin(BaseWindow):
    # [CLASS DOC]
    # Mixin deriving from the base window class that implements the statistic
    # methods shared by the rolling and expanding windows (it holds no state of
    # its own). Each statistic selects a compiled window kernel (or a numba
    # kernel when the numba engine is chosen) and runs it through the base
    # class's apply machinery. Subclasses (rolling, expanding) override most
    # methods only to attach their own documentation and call back into these
    # implementations.
    def count(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Number of non-missing observations in each window. numeric_only
        # (default false) restricts frames to numeric columns. Implemented as
        # a window sum of not-missing indicators (the base driver converts the
        # data to indicators for the count operation, which also lets
        # date-like data be counted). Returns float counts in the input's type.
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name="count", numeric_only=numeric_only)

    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC]
        # Applies a custom function to each window. func maps window values
        # to a single value; raw (default false, must be a boolean) chooses
        # between passing arrays (true) and series indexed by the on-axis
        # (false); engine (default none) selects cython or numba (none follows
        # the global numba option); engine_kwargs (default none) are numba jit
        # flags and are rejected for cython; args (default empty) and kwargs
        # (default empty) are extra arguments for func.
        # Numba path: requires raw true; function arguments are prepared (the
        # function must accept the window as its first argument), and either
        # the single-column or the table-wise compiled kernel is generated
        # according to the method. Cython path: builds a wrapper around the
        # compiled apply kernel. Raises a value error for a non-boolean raw, a
        # false raw with numba, engine_kwargs with cython, or an unknown
        # engine. Returns a result in the input's type via the base driver.
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        if not is_bool(raw):
            raise ValueError("raw parameter must be `True` or `False`")

        numba_args: tuple[Any, ...] = ()
        if maybe_use_numba(engine):
            if raw is False:
                raise ValueError("raw must be `True` when using the numba engine")
            numba_args, kwargs = prepare_function_arguments(
                func, args, kwargs, num_required_args=1
            )
            if self.method == "single":
                apply_func = generate_numba_apply_func(
                    func, **get_jit_arguments(engine_kwargs)
                )
            else:
                apply_func = generate_numba_table_func(
                    func, **get_jit_arguments(engine_kwargs)
                )
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")
            apply_func = self._generate_cython_apply_func(args, kwargs, raw, func)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

        return self._apply(
            apply_func,
            name="apply",
            numba_args=numba_args,
        )

    def _generate_cython_apply_func(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        raw: bool | np.bool_,
        function: Callable[..., Any],
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
        # [FUNCTION DOC]
        # HELPER: builds the kernel handed to the apply driver for the cython
        # engine. args and kwargs are extra arguments for the user function,
        # raw says whether to pass arrays, function is the user function.
        # Returns a callable taking values, window starts, ends and min
        # periods; when raw is false it first wraps the values in a series
        # indexed by the on-axis (without copying) so the function sees
        # labels, then invokes the compiled apply kernel.
        from pandas import Series

        window_func = partial(
            window_aggregations.roll_apply,
            args=args,
            kwargs=kwargs,
            raw=bool(raw),
            function=function,
        )

        def apply_func(values, begin, end, min_periods, raw=raw):
            # [FUNCTION DOC]
            # HELPER: the returned kernel: values (float array), begin and
            # end window bounds, min_periods; raw defaults to the outer raw
            # flag. Converts values to an on-axis-indexed series when raw is
            # false, then returns the compiled apply kernel's result array.
            if not raw:
                # GH 45912
                values = Series(values, index=self._on, copy=False)
            return window_func(values, begin, end, min_periods)

        return apply_func

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    # [FUNCTION DOC]
    # Overload stub (callable form): see the implementation below.

    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
    # [FUNCTION DOC]
    # Overload stub (callable plus data keyword form): see the implementation
    # below.

    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Passes this window object through a function. func is a callable
        # taking the window as its first argument, or a pair of (callable,
        # keyword name) so the window is passed by keyword; extra arguments
        # are forwarded. Delegates to the shared pipe helper (which raises a
        # value error if the keyword name is also supplied in the keyword
        # arguments) and returns whatever func returns.
        return com.pipe(self, func, *args, **kwargs)

    def sum(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Window sum. numeric_only (default false) restricts frames to numeric
        # columns; engine (default none) selects cython or numba (none follows
        # the global numba option); engine_kwargs (default none) are numba jit
        # flags. With numba: method table uses a column-wise nan-aware sum
        # through the custom apply path, otherwise a sliding-sum numba kernel
        # via the shared executor. Without numba: the compiled rolling sum
        # kernel through the base driver (engine_kwargs are not checked on this
        # path). Returns the input's type.
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nansum)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_sum

                return self._numba_apply(sliding_sum, engine_kwargs)
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name="sum", numeric_only=numeric_only)

    def max(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Window maximum; same parameters and structure as the window sum,
        # using the nan-aware maximum for table mode, the sliding min/max numba
        # kernel (maximum mode) for single mode, and the compiled maximum
        # kernel without numba.
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmax)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_min_max

                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=True)
        window_func = window_aggregations.roll_max
        return self._apply(window_func, name="max", numeric_only=numeric_only)

    def min(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Window minimum; same parameters and structure as the window sum,
        # using the nan-aware minimum for table mode, the sliding min/max numba
        # kernel (minimum mode) for single mode, and the compiled minimum
        # kernel without numba.
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmin)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_min_max

                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=False)
        window_func = window_aggregations.roll_min
        return self._apply(window_func, name="min", numeric_only=numeric_only)

    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Window mean; same parameters and structure as the window sum, using
        # the nan-aware mean for table mode, the sliding-mean numba kernel for
        # single mode, and the compiled mean kernel without numba.
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmean)
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                from pandas.core._numba.kernels import sliding_mean

                return self._numba_apply(sliding_mean, engine_kwargs)
        window_func = window_aggregations.roll_mean
        return self._apply(window_func, name="mean", numeric_only=numeric_only)

    def median(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Window median. Same parameters as the window sum. With numba, always
        # goes through the custom apply path with raw arrays, using the
        # nan-aware median (wrapped for column-wise reduction in table mode);
        # without numba uses the compiled median kernel. Returns the input's
        # type.
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmedian)
            else:
                func = np.nanmedian

            return self.apply(
                func,
                raw=True,
                engine=engine,
                engine_kwargs=engine_kwargs,
            )
        window_func = window_aggregations.roll_median_c
        return self._apply(window_func, name="median", numeric_only=numeric_only)

    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Window standard deviation. ddof (default one) is the delta degrees
        # of freedom; numeric_only, engine, engine_kwargs as for the window
        # sum. With numba: raises a not-implemented error for table mode,
        # otherwise the square root (clamping negatives to zero) of the sliding
        # variance kernel. Without numba: wraps the compiled variance kernel
        # with the safe square root and runs it through the base driver.
        if maybe_use_numba(engine):
            if self.method == "table":
                raise NotImplementedError("std not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var

            return zsqrt(self._numba_apply(sliding_var, engine_kwargs, ddof=ddof))
        window_func = window_aggregations.roll_var

        def zsqrt_func(values, begin, end, min_periods):
            # [FUNCTION DOC]
            # HELPER: kernel wrapper for the cython standard deviation: computes
            # the compiled windowed variance (with the captured delta degrees of
            # freedom) and returns its safe element-wise square root.
            return zsqrt(window_func(values, begin, end, min_periods, ddof=ddof))

        return self._apply(
            zsqrt_func,
            name="std",
            numeric_only=numeric_only,
        )

    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Window variance; same parameters as the window standard deviation.
        # With numba: not-implemented error for table mode, otherwise the
        # sliding variance kernel with ddof. Without numba: the compiled
        # variance kernel with ddof bound, through the base driver.
        if maybe_use_numba(engine):
            if self.method == "table":
                raise NotImplementedError("var not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var

            return self._numba_apply(sliding_var, engine_kwargs, ddof=ddof)
        window_func = partial(window_aggregations.roll_var, ddof=ddof)
        return self._apply(
            window_func,
            name="var",
            numeric_only=numeric_only,
        )

    def skew(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Window unbiased skewness using the compiled skew kernel through the
        # base driver. numeric_only (default false) restricts frames to numeric
        # columns. No numba engine option.
        window_func = window_aggregations.roll_skew
        return self._apply(
            window_func,
            name="skew",
            numeric_only=numeric_only,
        )

    def sem(self, ddof: int = 1, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Window standard error of the mean. ddof (default one) is the delta
        # degrees of freedom; numeric_only (default false). First validates the
        # numeric-only flag so errors name this operation, then returns the
        # window standard deviation divided by the square root of the window
        # count of non-missing values.
        # Raise here so error message says sem instead of std
        self._validate_numeric_only("sem", numeric_only)
        return self.std(numeric_only=numeric_only, ddof=ddof) / (
            self.count(numeric_only=numeric_only)
        ).pow(0.5)

    def kurt(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Window unbiased excess kurtosis using the compiled kurtosis kernel
        # through the base driver. numeric_only (default false). No numba
        # engine option.
        window_func = window_aggregations.roll_kurt
        return self._apply(
            window_func,
            name="kurt",
            numeric_only=numeric_only,
        )

    def first(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # First (left-most) value in each window using the compiled first
        # kernel through the base driver. numeric_only (default false).
        window_func = window_aggregations.roll_first
        return self._apply(
            window_func,
            name="first",
            numeric_only=numeric_only,
        )

    def last(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Last (right-most) value in each window using the compiled last
        # kernel through the base driver. numeric_only (default false).
        window_func = window_aggregations.roll_last
        return self._apply(
            window_func,
            name="last",
            numeric_only=numeric_only,
        )

    def quantile(
        self,
        q: float,
        interpolation: QuantileInterpolation = "linear",
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Window quantile. q is the quantile in zero to one; interpolation
        # (default linear) is linear, lower, higher, midpoint or nearest;
        # numeric_only (default false). A quantile of exactly one uses the
        # window maximum kernel and exactly zero uses the minimum kernel
        # (interpolation irrelevant); otherwise the compiled quantile kernel
        # with q and the interpolation mode bound. No range validation of q is
        # done here.
        if q == 1.0:
            window_func = window_aggregations.roll_max
        elif q == 0.0:
            window_func = window_aggregations.roll_min
        else:
            window_func = partial(
                window_aggregations.roll_quantile,
                quantile=q,
                interpolation=interpolation,
            )

        return self._apply(window_func, name="quantile", numeric_only=numeric_only)

    def rank(
        self,
        method: WindowingRankType = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Rank of the newest value within each window. method (default
        # average) is average, min or max for ties; ascending (default true)
        # sets ranking direction; pct (default false) returns percentile
        # ranks; numeric_only (default false). Uses the compiled rank kernel
        # with those options bound, through the base driver.
        window_func = partial(
            window_aggregations.roll_rank,
            method=method,
            ascending=ascending,
            percentile=pct,
        )

        return self._apply(window_func, name="rank", numeric_only=numeric_only)

    def nunique(
        self,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Number of distinct values in each window using the compiled distinct
        # count kernel through the base driver. numeric_only (default false).
        window_func = partial(
            window_aggregations.roll_nunique,
        )

        return self._apply(window_func, name="nunique", numeric_only=numeric_only)

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Window sample covariance. other (default none: use this object) is
        # the second series/frame; pairwise (default none) picks matching
        # columns versus all pairs; ddof (default one) is the delta degrees of
        # freedom; numeric_only (default false). Raises a not-implemented
        # error if a step is set or the numeric-only flag is invalid for the
        # data. The per-pair function computes window bounds, then the
        # covariance as the mean of products minus product of means, scaled by
        # count divided by count minus ddof, with floating point warnings
        # suppressed; counts use pairs where both values are present. The pair
        # function result is a series with the first object's index and name.
        # The combination over columns is delegated to the pairwise driver.
        if self.step is not None:
            raise NotImplementedError("step not implemented for cov")
        self._validate_numeric_only("cov", numeric_only)

        from pandas import Series

        def cov_func(x, y):
            # [FUNCTION DOC]
            # HELPER: nested per-pair function for the covariance. x and y are
            # two series; returns a series (index and name of x) of rolling
            # covariances as described by the enclosing method.
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            self._check_window_bounds(start, end, len(x_array))

            with np.errstate(all="ignore"):
                mean_x_y = window_aggregations.roll_mean(
                    x_array * y_array, start, end, min_periods
                )
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(
                    notna(x_array + y_array).astype(np.float64), start, end, 0
                )
                result = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, cov_func, numeric_only
        )

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Window Pearson correlation; same parameters, restrictions and
        # structure as the covariance. The per-pair function divides the
        # covariance numerator by the square root of the product of the
        # windowed variances of both inputs (computed with the same ddof).
        # Windows with zero variance give missing results from the zero by
        # zero division.
        if self.step is not None:
            raise NotImplementedError("step not implemented for corr")
        self._validate_numeric_only("corr", numeric_only)

        from pandas import Series

        def corr_func(x, y):
            # [FUNCTION DOC]
            # HELPER: nested per-pair function for the correlation. x and y
            # are two series; returns a series (index and name of x) of
            # rolling correlations as described by the enclosing method.
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            self._check_window_bounds(start, end, len(x_array))

            with np.errstate(all="ignore"):
                mean_x_y = window_aggregations.roll_mean(
                    x_array * y_array, start, end, min_periods
                )
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(
                    notna(x_array + y_array).astype(np.float64), start, end, 0
                )
                x_var = window_aggregations.roll_var(
                    x_array, start, end, min_periods, ddof
                )
                y_var = window_aggregations.roll_var(
                    y_array, start, end, min_periods, ddof
                )
                numerator = (mean_x_y - mean_x * mean_y) * (
                    count_x_y / (count_x_y - ddof)
                )
                denominator = (x_var * y_var) ** 0.5
                result = numerator / denominator
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, corr_func, numeric_only
        )


@set_module("pandas.api.typing")
class Rolling(RollingAndExpandingMixin):
    # [CLASS DOC]
    # Rolling window over a series or data frame, with windows defined by a
    # fixed number of observations, a time span (offset) on a datetime-like
    # axis, or a custom indexer. Subclass of the rolling/expanding mixin; holds
    # the base window state (see the base class) and adds validation specific
    # to rolling windows. The class docstring is assigned at module level from
    # the weighted window class, so it is shared with that class.
    # Validation rules: a time-offset window requires a datetime-like (or
    # empty/arrow time-typed) axis that is monotonic and free of missing
    # timestamps, converts the window to an integer duration in the axis's
    # unit (or period count units), defaults min periods to one, and forbids a
    # step; a custom indexer is accepted as is; anything else must be a
    # non-negative integer.
    # Methods: aggregate, count, apply, pipe and all statistics; most are thin
    # wrappers around the mixin that carry rolling-specific documentation.
    _attributes: list[str] = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "on",
        "closed",
        "step",
        "method",
    ]

    def _validate(self) -> None:
        # [FUNCTION DOC]
        # HELPER: extends the base validation for rolling windows; returns
        # nothing. If the axis is datetime-like, period, time-delta, arrow
        # time-typed or the object is empty, and the window is a string, offset
        # or timedelta: checks the axis is monotonic without missing values
        # (value error otherwise), converts the window to an offset (value
        # error if it is not compatible with a datetime-like axis, e.g.
        # non-fixed frequencies), caches the window as an integer duration in
        # the axis's time unit (nanoseconds if undeterminable; for period axes,
        # the window measured in multiples of the period frequency), sets min
        # periods to one if unset, and raises a not-implemented error if a step
        # is given. A custom indexer is accepted without checks. Anything else
        # raises a value error unless it is an integer of zero or more.
        super()._validate()

        # we allow rolling on a datetimelike index
        if (
            self.obj.empty
            or isinstance(self._on, (DatetimeIndex, TimedeltaIndex, PeriodIndex))
            or (isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM")
        ) and isinstance(self.window, (str, BaseOffset, timedelta)):
            self._validate_datetimelike_monotonic()

            # this will raise ValueError on non-fixed freqs
            try:
                freq = to_offset(self.window)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"passed window {self.window} is not "
                    "compatible with a datetimelike index"
                ) from err
            if isinstance(self._on, PeriodIndex):
                # error: Incompatible types in assignment (expression has type
                # "float", variable has type "Optional[int]")
                self._win_freq_i8 = freq.nanos / (  # type: ignore[assignment]
                    self._on.freq.nanos / self._on.freq.n
                )
            else:
                try:
                    unit = dtype_to_unit(self._on.dtype)  # type: ignore[arg-type]
                except TypeError:
                    # if not a datetime dtype, eg for empty dataframes
                    unit = "ns"
                unit = cast("TimeUnit", unit)
                self._win_freq_i8 = Timedelta(freq.nanos).as_unit(unit)._value

            # min_periods must be an integer
            if self.min_periods is None:
                self.min_periods = 1

            if self.step is not None:
                raise NotImplementedError(
                    "step is not supported with frequency windows"
                )

        elif isinstance(self.window, BaseIndexer):
            # Passed BaseIndexer subclass should handle all other rolling kwargs
            pass
        elif not is_integer(self.window) or self.window < 0:
            raise ValueError("window must be an integer 0 or greater")

    def _validate_datetimelike_monotonic(self) -> None:
        # [FUNCTION DOC]
        # HELPER: for frequency windows, raises a value error if the on-axis
        # contains missing timestamps, or is neither monotonic increasing nor
        # monotonic decreasing. Returns nothing. Overridden in the groupby
        # variant to check each group.
        """
        Validate self._on is monotonic (increasing or decreasing) and has
        no NaT values for frequency windows.
        """
        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
        if not (self._on.is_monotonic_increasing or self._on.is_monotonic_decreasing):
            self._raise_monotonic_error("values must be monotonic")

    def _raise_monotonic_error(self, msg: str):
        # [FUNCTION DOC]
        # HELPER: always raises a value error whose message is the on option
        # (or the word index when unset) followed by msg.
        on = self.on
        if on is None:
            on = "index"
        raise ValueError(f"{on} {msg}")

    def aggregate(self, func=None, *args, **kwargs):
        # [FUNCTION DOC]
        # Aggregates the rolling windows with one or more functions (function,
        # name, list, or mapping; default none) plus extra arguments and
        # keyword arguments. Delegates entirely to the base implementation
        # (which supports relabeling and falls back to a custom apply) and
        # returns its result. Also exposed under the shorter alias name.
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data. If a function, must either
            work when passed a Series/Dataframe or
            when passed to Series/Dataframe.apply.

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
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrame data.

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
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
        A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2).sum()
            A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  5.0  11.0  17.0

        >>> df.rolling(2).agg({"A": "sum", "B": "min"})
            A    B
        0  NaN  NaN
        1  3.0  4.0
        2  5.0  5.0
        """
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate

    def count(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Rolling count of non-missing observations; numeric_only (default
        # false) restricts frames to numeric columns. Delegates to the mixin
        # (passing the flag positionally) and returns float counts in the
        # input's type.
        """
        Calculate the rolling count of non NaN observations.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.count : Aggregating count for Series.
        DataFrame.count : Aggregating count for DataFrame.

        Examples
        --------
        >>> s = pd.Series([2, 3, np.nan, 10])
        >>> s.rolling(2).count()
        0    NaN
        1    2.0
        2    1.0
        3    1.0
        dtype: float64
        >>> s.rolling(3).count()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        dtype: float64
        >>> s.rolling(4).count()
        0    NaN
        1    NaN
        2    NaN
        3    3.0
        dtype: float64
        """
        return super().count(numeric_only)

    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC]
        # Rolling custom aggregation. Parameters as in the mixin: func,
        # raw (default false), engine (default none), engine_kwargs (default
        # none), args and kwargs (default none, treated as empty). Delegates to
        # the mixin, which validates them and returns the input's type.
        """
        Calculate the rolling custom aggregation function.

        Parameters
        ----------
        func : function
            Must produce a single value from an ndarray input if ``raw=True``
            or a single value from a Series if ``raw=False``. Can also accept a
            Numba JIT function with ``engine='numba'`` specified.

        raw : bool, default False
            * ``False`` : passes each row or column as a Series to the
              function.
            * ``True`` : the passed function will receive ndarray
              objects instead.

            If you are just applying a NumPy reduction function this will
            achieve much better performance.

        engine : str, default None
            * ``'cython'`` : Runs rolling apply through C-extensions from cython.
            * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
              Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``.

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``.

            The default ``engine_kwargs`` for the ``'numba'`` engine is
            ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
            applied to both the ``func`` and the ``apply`` rolling aggregation.

        args : tuple, default None
            Positional arguments to be passed into func.

        kwargs : dict, default None
            Keyword arguments to be passed into func.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.apply : Aggregating apply for Series.
        DataFrame.apply : Aggregating apply for DataFrame.

        Examples
        --------
        >>> ser = pd.Series([1, 6, 5, 4])
        >>> ser.rolling(2).apply(lambda s: s.sum() - s.min())
        0    NaN
        1    6.0
        2    6.0
        3    5.0
        dtype: float64
        """
        return super().apply(
            func,
            raw=raw,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    # [FUNCTION DOC]
    # Overload stub (callable form): see the final implementation below.

    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
    # [FUNCTION DOC]
    # Overload stub (callable plus data keyword form): see the final
    # implementation below.

    @final
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        # [FUNCTION DOC]
        # Implementation of the overloaded pipe (final). func is a callable
        # receiving this window object first, or a pair of (callable, keyword
        # name) to pass it by keyword; extra arguments are forwarded. Delegates
        # to the mixin and returns what func returns.
        """
        Apply a ``func`` with arguments to this Rolling object and return its result.

        Use `.pipe` when you want to improve readability by chaining together
        functions that expect
        Series, DataFrames, GroupBy, Rolling, Expanding or Resampler
        objects.
        Instead of writing

        >>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
        >>> g = lambda x, arg1: x * 5 / arg1
        >>> f = lambda x: x**4
        >>> df = pd.DataFrame(
        ...     {"A": [1, 2, 3, 4]}, index=pd.date_range("2012-08-02", periods=4)
        ... )
        >>> h(g(f(df.rolling("2D")), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP

        You can write

        >>> (
        ...     df.rolling("2D").pipe(f).pipe(g, arg1=1).pipe(h, arg2=2, arg3=3)
        ... )  # doctest: +SKIP

        which is much more readable.

        Parameters
        ----------
        func : callable or tuple of (callable, str)
            Function to apply to this Rolling object or, alternatively,
            a `(callable, data_keyword)` tuple where `data_keyword` is a
            string indicating the keyword of `callable` that expects the
            Rolling object.
        *args : iterable, optional
            Positional arguments passed into `func`.
        **kwargs : dict, optional
                A dictionary of keyword arguments passed into `func`.

        Returns
        -------
        Rolling
            The original object with the function `func` applied.

        See Also
        --------
        Series.pipe : Apply a function with arguments to a series.
        DataFrame.pipe: Apply a function with arguments to a dataframe.
        apply : Apply function to each group instead of to the
            full Rolling object.

        Notes
        -----
        See more `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`__.

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

        To get the difference between each rolling
        2-day window's maximum and minimum
        value in one pass, you can do

        >>> df.rolling("2D").pipe(lambda x: x.max() - x.min())
                    A
        2012-08-02  0.0
        2012-08-03  1.0
        2012-08-04  1.0
        2012-08-05  1.0
        """
        return super().pipe(func, *args, **kwargs)

    def sum(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Rolling sum; parameters numeric_only (default false), engine
        # (default none) and engine_kwargs (default none) are forwarded to the
        # mixin by keyword. Returns the input's type with float results.
        """
        Calculate the rolling sum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}``.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.sum : Aggregating sum for Series.
        DataFrame.sum : Aggregating sum for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba`
        for extended documentation and performance considerations
        for the Numba engine.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> s
        0    1
        1    2
        2    3
        3    4
        4    5
        dtype: int64

        >>> s.rolling(3).sum()
        0     NaN
        1     NaN
        2     6.0
        3     9.0
        4    12.0
        dtype: float64

        >>> s.rolling(3, center=True).sum()
        0     NaN
        1     6.0
        2     9.0
        3    12.0
        4     NaN
        dtype: float64

        For DataFrame, each sum is computed column-wise.

        >>> df = pd.DataFrame({"A": s, "B": s**2})
        >>> df
        A   B
        0  1   1
        1  2   4
        2  3   9
        3  4  16
        4  5  25

        >>> df.rolling(3).sum()
             A     B
        0   NaN   NaN
        1   NaN   NaN
        2   6.0  14.0
        3   9.0  29.0
        4  12.0  50.0
        """
        return super().sum(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def max(
        self,
        numeric_only: bool = False,
        *args,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Rolling maximum; numeric_only (default false), engine (default
        # none), engine_kwargs (default none) are forwarded to the mixin.
        # KNOWN ISSUE: unlike its sibling methods, the signature also accepts
        # extra positional and keyword arguments, which are silently ignored
        # and never forwarded.
        """
        Calculate the rolling maximum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        *args : iterable, optional
            Positional arguments passed into ``func``.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``.

            The default ``engine_kwargs`` for the ``'numba'`` engine is
            ``{'nopython': True, 'nogil': False, 'parallel': False}``.

        **kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.max : Aggregating max for Series.
        DataFrame.max : Aggregating max for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba`
        for extended documentation and performance considerations
        for the Numba engine.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.rolling(2).max()
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        dtype: float64
        """
        return super().max(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def min(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Rolling minimum; numeric_only (default false), engine (default
        # none), engine_kwargs (default none) are forwarded to the mixin.
        """
        Calculate the rolling minimum.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``.

            The default ``engine_kwargs`` for the ``'numba'`` engine is
            ``{'nopython': True, 'nogil': False, 'parallel': False}``.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.min : Aggregating min for Series.
        DataFrame.min : Aggregating min for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba`
        for extended documentation and performance considerations
        for the Numba engine.

        Examples
        --------
        Performing a rolling minimum with a window size of 3.

        >>> s = pd.Series([4, 3, 5, 2, 6])
        >>> s.rolling(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        dtype: float64
        """
        return super().min(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Rolling mean; numeric_only (default false), engine (default none),
        # engine_kwargs (default none) are forwarded to the mixin.
        """
        Calculate the rolling mean.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``.

            The default ``engine_kwargs`` for the ``'numba'`` engine is
            ``{'nopython': True, 'nogil': False, 'parallel': False}``.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.mean : Aggregating mean for Series.
        DataFrame.mean : Aggregating mean for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba`
        for extended documentation and performance considerations
        for the Numba engine.

        Examples
        --------
        The below examples will show rolling mean calculations with window sizes of
        two and three, respectively.

        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).mean()
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64

        >>> s.rolling(3).mean()
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        dtype: float64
        """
        return super().mean(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def median(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Rolling median; numeric_only (default false), engine (default none),
        # engine_kwargs (default none) are forwarded to the mixin.
        """
        Calculate the rolling median.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``.

            The default ``engine_kwargs`` for the ``'numba'`` engine is
            ``{'nopython': True, 'nogil': False, 'parallel': False}``.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.median : Aggregating median for Series.
        DataFrame.median : Aggregating median for DataFrame.

        Notes
        -----
        See :ref:`window.numba_engine` and :ref:`enhancingperf.numba`
        for extended documentation and performance considerations
        for the Numba engine.

        Examples
        --------
        Compute the rolling median of a series with a window size of 3.

        >>> s = pd.Series([0, 1, 2, 3, 4])
        >>> s.rolling(3).median()
        0    NaN
        1    NaN
        2    1.0
        3    2.0
        4    3.0
        dtype: float64
        """
        return super().median(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Rolling standard deviation; ddof (default one) delta degrees of
        # freedom, numeric_only (default false), engine and engine_kwargs
        # (default none) are forwarded to the mixin.
        """
        Calculate the rolling standard deviation.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``.

            The default ``engine_kwargs`` for the ``'numba'`` engine is
            ``{'nopython': True, 'nogil': False, 'parallel': False}``.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        numpy.std : Equivalent method for NumPy array.
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.std : Aggregating std for Series.
        DataFrame.std : Aggregating std for DataFrame.

        Notes
        -----
        The default ``ddof`` of 1 used in :meth:`Series.std` is different
        than the default ``ddof`` of 0 in :func:`numpy.std`.

        A minimum of one period is required for the rolling calculation.

        Examples
        --------
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    1.000000
        4    1.000000
        5    1.154701
        6    0.000000
        dtype: float64
        """
        return super().std(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # [FUNCTION DOC]
        # Rolling variance; same parameters as the rolling standard deviation,
        # forwarded to the mixin.
        """
        Calculate the rolling variance.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or
              globally setting ``compute.use_numba``

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``.

            The default ``engine_kwargs`` for the ``'numba'`` engine is
            ``{'nopython': True, 'nogil': False, 'parallel': False}``.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        numpy.var : Equivalent method for NumPy array.
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.var : Aggregating var for Series.
        DataFrame.var : Aggregating var for DataFrame.

        Notes
        -----
        The default ``ddof`` of 1 used in :meth:`Series.var` is different
        than the default ``ddof`` of 0 in :func:`numpy.var`.

        A minimum of one period is required for the rolling calculation.

        Examples
        --------
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    1.000000
        4    1.000000
        5    1.333333
        6    0.000000
        dtype: float64
        """
        return super().var(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    def skew(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Rolling unbiased skewness (needs at least three observations per
        # window); numeric_only (default false). Delegates to the mixin.
        """
        Calculate the rolling unbiased skewness.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        scipy.stats.skew : Third moment of a probability density.
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.skew : Aggregating skew for Series.
        DataFrame.skew : Aggregating skew for DataFrame.

        Notes
        -----

        A minimum of three periods is required for the rolling calculation.

        Examples
        --------
        >>> ser = pd.Series([1, 5, 2, 7, 15, 6])
        >>> ser.rolling(3).skew().round(6)
        0         NaN
        1         NaN
        2    1.293343
        3   -0.585583
        4    0.670284
        5    1.652317
        dtype: float64
        """
        return super().skew(numeric_only=numeric_only)

    def sem(self, ddof: int = 1, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Rolling standard error of the mean; ddof (default one),
        # numeric_only (default false). Validates the numeric-only flag first
        # (so errors name this operation), then divides the rolling standard
        # deviation by the square root of the rolling count. Does not call the
        # mixin's version (it repeats the same logic here).
        """
        Calculate the rolling standard error of mean.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.sem : Aggregating sem for Series.
        DataFrame.sem : Aggregating sem for DataFrame.

        Notes
        -----
        A minimum of one period is required for the calculation.

        Examples
        --------
        >>> s = pd.Series([0, 1, 2, 3])
        >>> s.rolling(2, min_periods=1).sem()
        0    NaN
        1    0.5
        2    0.5
        3    0.5
        dtype: float64
        """
        # Raise here so error message says sem instead of std
        self._validate_numeric_only("sem", numeric_only)
        return self.std(numeric_only=numeric_only, ddof=ddof) / (
            self.count(numeric_only)
        ).pow(0.5)

    def kurt(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Rolling unbiased excess kurtosis (needs at least four observations
        # per window); numeric_only (default false). Delegates to the mixin.
        """
        Calculate the rolling Fisher's definition of kurtosis without bias.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        scipy.stats.kurtosis : Reference SciPy method.
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.kurt : Aggregating kurt for Series.
        DataFrame.kurt : Aggregating kurt for DataFrame.

        Notes
        -----
        A minimum of four periods is required for the calculation.

        Examples
        --------
        The example below will show a rolling calculation with a window size of
        four matching the equivalent function call using `scipy.stats`.

        >>> arr = [1, 2, 3, 4, 999]
        >>> import scipy.stats
        >>> print(f"{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}")
        -1.200000
        >>> print(f"{scipy.stats.kurtosis(arr[1:], bias=False):.6f}")
        3.999946
        >>> s = pd.Series(arr)
        >>> s.rolling(4).kurt()
        0         NaN
        1         NaN
        2         NaN
        3   -1.200000
        4    3.999946
        dtype: float64
        """
        return super().kurt(numeric_only=numeric_only)

    def first(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # First (left-most) element of each rolling window; numeric_only
        # (default false). Delegates to the mixin.
        """
        Calculate the rolling First (left-most) element of the window.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        GroupBy.first : Similar method for GroupBy objects.
        Rolling.last : Method to get the last element in each window.

        Examples
        --------
        The example below will show a rolling calculation with a window size of
        three.

        >>> s = pd.Series(range(5))
        >>> s.rolling(3).first()
        0         NaN
        1         NaN
        2         0.0
        3         1.0
        4         2.0
        dtype: float64
        """
        return super().first(numeric_only=numeric_only)

    def last(self, numeric_only: bool = False):
        # [FUNCTION DOC]
        # Last (right-most) element of each rolling window; numeric_only
        # (default false). Delegates to the mixin.
        """
        Calculate the rolling Last (right-most) element of the window.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        GroupBy.last : Similar method for GroupBy objects.
        Rolling.first : Method to get the first element in each window.

        Examples
        --------
        The example below will show a rolling calculation with a window size of
        three.

        >>> s = pd.Series(range(5))
        >>> s.rolling(3).last()
        0         NaN
        1         NaN
        2         2.0
        3         3.0
        4         4.0
        dtype: float64
        """
        return super().last(numeric_only=numeric_only)

    def quantile(
        self,
        q: float,
        interpolation: QuantileInterpolation = "linear",
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Rolling quantile; q is the quantile in zero to one, interpolation
        # (default linear) one of linear, lower, higher, midpoint, nearest,
        # numeric_only (default false). Delegates to the mixin.
        """
        Calculate the rolling quantile.

        Parameters
        ----------
        q : float
            Quantile to compute. 0 <= quantile <= 1.

        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.quantile : Aggregating quantile for Series.
        DataFrame.quantile : Aggregating quantile for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).quantile(0.4, interpolation="lower")
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        dtype: float64

        >>> s.rolling(2).quantile(0.4, interpolation="midpoint")
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64
        """
        return super().quantile(
            q=q,
            interpolation=interpolation,
            numeric_only=numeric_only,
        )

    def rank(
        self,
        method: WindowingRankType = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Rolling rank of the newest value in each window; method (default
        # average) for ties, ascending (default true), pct (default false) for
        # percentile ranks, numeric_only (default false). Delegates to the
        # mixin.
        """
        Calculate the rolling rank.

        Parameters
        ----------
        method : {'average', 'min', 'max'}, default 'average'
            How to rank the group of records that have the same value (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group

        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.

        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.rank : Aggregating rank for Series.
        DataFrame.rank : Aggregating rank for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 4, 2, 3, 5, 3])
        >>> s.rolling(3).rank()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.5
        dtype: float64

        >>> s.rolling(3).rank(method="max")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    2.0
        dtype: float64

        >>> s.rolling(3).rank(method="min")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.0
        dtype: float64
        """
        return super().rank(
            method=method,
            ascending=ascending,
            pct=pct,
            numeric_only=numeric_only,
        )

    def nunique(
        self,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Rolling count of distinct values; numeric_only (default false).
        # Delegates to the mixin.
        """
        Calculate the rolling nunique.

        .. versionadded:: 3.0.0

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.nunique : Aggregating nunique for Series.
        DataFrame.nunique : Aggregating nunique for DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 4, 2, np.nan, 3, 3, 4, 5])
        >>> s.rolling(3).nunique()
        0    NaN
        1    NaN
        2    3.0
        3    NaN
        4    NaN
        5    NaN
        6    2.0
        7    3.0
        dtype: float64
        """
        return super().nunique(
            numeric_only=numeric_only,
        )

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Rolling sample covariance; other (default none), pairwise (default
        # none), ddof (default one), numeric_only (default false). Delegates
        # to the mixin, which raises a not-implemented error when a step is
        # set.
        """
        Calculate the rolling sample covariance.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.

        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.

        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.cov : Aggregating cov for Series.
        DataFrame.cov : Aggregating cov for DataFrame.

        Examples
        --------
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([1, 4, 5, 8])
        >>> ser1.rolling(2).cov(ser2)
        0    NaN
        1    1.5
        2    0.5
        3    1.5
        dtype: float64
        """
        return super().cov(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # [FUNCTION DOC]
        # Rolling Pearson correlation; same parameters as the rolling
        # covariance. Delegates to the mixin.
        """
        Calculate the rolling correlation.

        Parameters
        ----------
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.

        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.

        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        numeric_only : bool, default False
            Include only float, int, boolean columns.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object with ``np.float64`` dtype.

        See Also
        --------
        cov : Similar method to calculate covariance.
        numpy.corrcoef : NumPy Pearson's correlation calculation.
        Series.rolling : Calling rolling with Series data.
        DataFrame.rolling : Calling rolling with DataFrames.
        Series.corr : Aggregating corr for Series.
        DataFrame.corr : Aggregating corr for DataFrame.

        Notes
        -----
        This function uses Pearson's definition of correlation
        (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

        When `other` is not specified, the output will be self correlation (e.g.
        all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`
        set to `True`.

        Function will return ``NaN`` for correlations of equal valued sequences;
        this is the result of a 0/0 division error.

        When `pairwise` is set to `False`, only matching columns between `self` and
        `other` will be used.

        When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame
        with the original index on the first level, and the `other` DataFrame
        columns on the second level.

        In the case of missing elements, only complete pairwise observations
        will be used.

        Examples
        --------
        The below example shows a rolling calculation with a window size of
        four matching the equivalent function call using :meth:`numpy.corrcoef`.

        >>> v1 = [3, 3, 3, 5, 8]
        >>> v2 = [3, 4, 4, 4, 8]
        >>> np.corrcoef(v1[:-1], v2[:-1])
        array([[1.        , 0.33333333],
            [0.33333333, 1.        ]])
        >>> np.corrcoef(v1[1:], v2[1:])
        array([[1.       , 0.9169493],
            [0.9169493, 1.       ]])
        >>> s1 = pd.Series(v1)
        >>> s2 = pd.Series(v2)
        >>> s1.rolling(4).corr(s2)
        0         NaN
        1         NaN
        2         NaN
        3    0.333333
        4    0.916949
        dtype: float64

        The below example shows a similar rolling calculation on a
        DataFrame using the pairwise option.

        >>> matrix = np.array(
        ...     [[51.0, 35.0], [49.0, 30.0], [47.0, 32.0], [46.0, 31.0], [50.0, 36.0]]
        ... )
        >>> np.corrcoef(matrix[:-1, 0], matrix[:-1, 1])
        array([[1.       , 0.6263001],
            [0.6263001, 1.       ]])
        >>> np.corrcoef(matrix[1:, 0], matrix[1:, 1])
        array([[1.        , 0.55536811],
            [0.55536811, 1.        ]])
        >>> df = pd.DataFrame(matrix, columns=["X", "Y"])
        >>> df
              X     Y
        0  51.0  35.0
        1  49.0  30.0
        2  47.0  32.0
        3  46.0  31.0
        4  50.0  36.0
        >>> df.rolling(4).corr(pairwise=True)
                    X         Y
        0 X        NaN       NaN
          Y        NaN       NaN
        1 X        NaN       NaN
          Y        NaN       NaN
        2 X        NaN       NaN
          Y        NaN       NaN
        3 X   1.000000  0.626300
          Y   0.626300  1.000000
        4 X   1.000000  0.555368
          Y   0.555368  1.000000
        """
        return super().corr(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )


Rolling.__doc__ = Window.__doc__


@set_module("pandas.api.typing")
class RollingGroupby(BaseWindowGroupby, Rolling):
    # [CLASS DOC]
    # Rolling window computed independently within each group of a groupby
    # object: combines the groupby window base (group-ordering and result index
    # reconstruction) with the rolling class (statistics and validation). The
    # attribute list concatenates both parents'. Overrides the window indexer
    # hook and the monotonic validation so both operate per group.
    """
    Provide a rolling groupby implementation.
    """

    _attributes = Rolling._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        # [FUNCTION DOC]
        # HELPER: builds a groupby-aware indexer that applies a rolling
        # indexer inside each group. Chooses the inner indexer class and window
        # size: for a custom indexer window, its own class, with its attributes
        # copied as constructor keyword arguments (minus any stored index
        # array, since each group's own axis is used); for a cached time
        # offset window, the variable indexer with that duration; otherwise
        # the fixed indexer with the window size. Combines it with the on-axis
        # integer array, the grouper's group-to-positions mapping and returns
        # the groupby indexer.
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        rolling_indexer: type[BaseIndexer]
        indexer_kwargs: dict[str, Any] | None = None
        index_array = self._index_array
        if isinstance(self.window, BaseIndexer):
            rolling_indexer = type(self.window)
            indexer_kwargs = self.window.__dict__.copy()
            assert isinstance(indexer_kwargs, dict)  # for mypy
            # We'll be using the index of each group later
            indexer_kwargs.pop("index_array", None)
            window = self.window
        elif self._win_freq_i8 is not None:
            rolling_indexer = VariableWindowIndexer
            # error: Incompatible types in assignment (expression has type
            # "int", variable has type "BaseIndexer")
            window = self._win_freq_i8  # type: ignore[assignment]
        else:
            rolling_indexer = FixedWindowIndexer
            window = self.window
        window_indexer = GroupbyIndexer(
            index_array=index_array,
            window_size=window,
            groupby_indices=self._grouper.indices,
            window_indexer=rolling_indexer,
            indexer_kwargs=indexer_kwargs,
        )
        return window_indexer

    def _validate_datetimelike_monotonic(self) -> None:
        # [FUNCTION DOC]
        # HELPER: per-group version of the monotonic check. Raises a value
        # error if the on-axis has missing timestamps; for each group, takes
        # that group's axis values and raises a value error (naming the on
        # option or index and asking to sort first) unless they are monotonic
        # increasing or decreasing. Returns nothing. Overall order across
        # groups is not required.
        """
        Validate that each group in self._on is monotonic
        """
        # GH 46061
        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
        for group_indices in self._grouper.indices.values():
            group_on = self._on.take(group_indices)
            if not (
                group_on.is_monotonic_increasing or group_on.is_monotonic_decreasing
            ):
                on = "index" if self.on is None else self.on
                raise ValueError(
                    f"Each group within {on} must be monotonic. "
                    f"Sort the values in {on} first."
                )
