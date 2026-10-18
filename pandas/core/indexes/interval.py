"""define the IntervalIndex"""

# [MODULE DOC]
# Implementation of IntervalIndex, the index of intervals that are all closed on
# the same side, and the fixed-frequency interval builder function.
# Public contents: the IntervalIndex class and the interval_range function (both
# re-labelled to the top-level package namespace); private helpers for computing
# the label just after or before a given label, for unpickling, and for the
# argument checks of the range builder.
# Relationships: data lives in an IntervalArray; the class subclasses the
# extension-array-backed index base. Lookups use a compiled interval tree engine
# instead of the usual hash engine. Properties such as the open/closed flags,
# overlaps, containment, closed-side change and conversion to tuples are pinned
# from the array at import time by decorators. Imports the multi-level index
# (to expose left and right as levels), the datetime and timedelta range
# builders and the shared conversion utilities. No import-time side effects
# beyond class decoration.

from __future__ import annotations

from operator import (
    le,
    lt,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.interval import (
    Interval,
    IntervalMixin,
    IntervalTree,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)
from pandas.util._exceptions import rewrite_exception

from pandas.core.dtypes.cast import (
    find_common_type,
    infer_dtype_from_scalar,
    maybe_box_datetimelike,
    maybe_downcast_numeric,
    maybe_unbox_numpy_scalar,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
    IntervalArray,
)
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
from pandas.core.indexes.base import (
    Index,
    ensure_index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.extension import (
    ExtensionIndex,
    inherit_names,
)
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        npt,
    )


def _get_next_label(label):
    # [FUNCTION DOC]
    # HELPER: returns the smallest representable label greater than a given label, so
    # open right bounds can be turned into closed ones during slicing. Parameter
    # label is a scalar. Its dtype is taken from the object or its type (timestamps
    # and timedeltas are treated as nanosecond datetimes). Datetime-like and
    # timedelta labels return the label plus one nanosecond; integers add one;
    # floats move to the next floating-point number toward positive infinity. Raises
    # a type error for any other type.
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label + np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label + 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {type(label)!r}")


def _get_prev_label(label):
    # [FUNCTION DOC]
    # HELPER: returns the largest representable label less than a given label, mirror
    # of the next-label helper: one nanosecond earlier for datetime-like, minus one
    # for integers, the preceding floating-point number for floats. Raises a type
    # error for other types.
    # KNOWN ISSUE: the error message says it cannot determine the next label, although this function computes the previous label (copy-paste from the sibling).
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label - np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label - 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, -np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {type(label)!r}")


def _new_IntervalIndex(cls, d):
    # [FUNCTION DOC]
    # HELPER: reconstructor used when unpickling. Parameters: cls is the class; d is
    # the dictionary of saved state (left, right, closed, name). Returns the result
    # of calling the arrays-based constructor with these as keywords.
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """
    return cls.from_arrays(**d)


@inherit_names(["set_closed", "to_tuples"], IntervalArray, wrap=True)
@inherit_names(
    [
        "__array__",
        "overlaps",
        "contains",
        "closed_left",
        "closed_right",
        "open_left",
        "open_right",
        "is_empty",
    ],
    IntervalArray,
)
@inherit_names(["is_non_overlapping_monotonic", "closed"], IntervalArray, cache=True)
@set_module("pandas")
class IntervalIndex(ExtensionIndex):
    # [CLASS DOC]
    # Immutable index of intervals sharing one closed side (left, right, both, or
    # neither). Inherits from the extension index base. State: _data is an
    # IntervalArray; left and right are cached Index views of its bounds; the closed
    # side and the open/closed flags come from the array. Equality-style lookup uses
    # a cached interval-tree engine rather than a hash table. Key members: three
    # alternative constructors (from breaks, arrays and tuples), a containment test
    # accepting only intervals (or a missing value), location lookup supporting
    # scalar points inside intervals, indexer computation (exact interval matching,
    # pointwise fallback for overlapping data), overlap checks, midpoint and length,
    # and specialized intersection. Arithmetic is not implemented (a note says so).
    """
    Immutable index of intervals that are closed on the same side.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing
        Interval objects from which to build the IntervalIndex.
    closed : {'left', 'right', 'both', 'neither'}, default 'right'
        Whether the intervals are closed on the left-side, right-side, both or
        neither.
    dtype : dtype or None, default None
        If None, dtype will be inferred.
    copy : bool, default None
        Whether to copy input data, only relevant for array, Series, and Index
        inputs (for other input, e.g. a list, a new array is created anyway).
        Defaults to True for array input and False for Index/Series.
        Set to False to avoid copying array input at your own risk (if you
        know the input data won't be modified elsewhere).
        Set to True to force copying Series/Index input up front.
    name : object, optional
         Name to be stored in the index.
    verify_integrity : bool, default True
        Verify that the IntervalIndex is valid.

    Attributes
    ----------
    left
    right
    closed
    mid
    length
    is_empty
    is_non_overlapping_monotonic
    is_overlapping
    values

    Methods
    -------
    from_arrays
    from_tuples
    from_breaks
    contains
    overlaps
    set_closed
    to_tuples

    See Also
    --------
    Index : The base pandas Index type.
    Interval : A bounded slice-like interval; the elements of an IntervalIndex.
    interval_range : Function to create a fixed frequency IntervalIndex.
    cut : Bin values into discrete Intervals.
    qcut : Bin values into equal-sized Intervals based on rank or sample quantiles.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__
    for more.

    Examples
    --------
    A new ``IntervalIndex`` is typically constructed using
    :func:`interval_range`:

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalIndex.from_arrays`,
    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.

    See further examples in the doc strings of ``interval_range`` and the
    mentioned constructor methods.
    """

    _typ = "intervalindex"

    # annotate properties pinned via inherit_names
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool

    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings = False
    _data_cls = IntervalArray

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
        name: Hashable | None = None,
        verify_integrity: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Construct an IntervalIndex. Parameters: data is a 1-d array-like of Interval
        # objects (or an interval array); closed is the closed side (default none, taken
        # from data or right); dtype (default none, inferred); copy tri-state flag
        # resolved with the shared array-input helper (GH#63388); name (default none,
        # taken from data); verify_integrity (default true) validates the bounds. Builds an
        # IntervalArray with errors rewritten to mention this class name instead of the
        # array class, then wraps it with the private constructor. Returns the index.
        name = maybe_extract_name(name, data, cls)

        # GH#63388
        data, copy = cls._maybe_copy_array_input(data, copy, dtype)

        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray(
                data,
                closed=closed,
                copy=copy,
                dtype=dtype,
                verify_integrity=verify_integrity,
            )

        return cls._simple_new(array, name)

    @classmethod
    def from_breaks(
        cls,
        breaks,
        closed: IntervalClosedType | None = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        # [FUNCTION DOC]
        # Class method: build from a 1-d sequence of break points, where consecutive
        # values form intervals. Parameters: breaks; closed (default right); name
        # (default none); copy (default false); dtype (default none). Errors from the
        # array are rewritten to mention this class. Returns a new IntervalIndex.
        """
        Construct an IntervalIndex from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        name : str, optional
            Name of the resulting IntervalIndex.
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalIndex.from_arrays : Construct from a left and right array.
        IntervalIndex.from_tuples : Construct from a sequence of tuples.

        Examples
        --------
        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_breaks(
                breaks, closed=closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        # [FUNCTION DOC]
        # Class method: build from separate arrays of left and right bounds. Parameters:
        # left, right (1-d array-likes); closed (default right); name (default none);
        # copy (default false); dtype (default none, inferred). Raises a value error when
        # only one of a pair is missing or when a left bound exceeds its right bound, and
        # a type error for unsupported subtypes (category, object, string); errors are
        # rewritten to mention this class. Returns a new IntervalIndex.
        """
        Construct from two arrays defining the left and right bounds.

        Parameters
        ----------
        left : array-like (1-dimensional)
            Left bounds for each interval.
        right : array-like (1-dimensional)
            Right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        name : str, optional
            Name of the resulting IntervalIndex.
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalIndex.from_breaks : Construct an IntervalIndex from an array of
            splits.
        IntervalIndex.from_tuples : Construct an IntervalIndex from an
            array-like of tuples.

        Notes
        -----
        Each element of `left` must be less than or equal to the `right`
        element at the same position. If an element is missing, it must be
        missing in both `left` and `right`. A TypeError is raised when
        using an unsupported type for `left` or `right`. At the moment,
        'category', 'object', and 'string' subtypes are not supported.

        Examples
        --------
        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_arrays(
                left, right, closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        # [FUNCTION DOC]
        # Class method: build from a sequence of two-element tuples. Parameters: data;
        # closed (default right); name (default none); copy (default false); dtype
        # (default none). Delegates to the array constructor (errors rewritten to this
        # class name) and returns a new IntervalIndex.
        # KNOWN ISSUE: the docstring says the copy argument is compat-only and ignored, but the value is forwarded to the array constructor.
        """
        Construct an IntervalIndex from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        name : str, optional
            Name of the resulting IntervalIndex.
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalIndex.from_arrays : Construct an IntervalIndex from a left and
                                    right array.
        IntervalIndex.from_breaks : Construct an IntervalIndex from an array of
                                    splits.

        Examples
        --------
        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])
        IntervalIndex([(0, 1], (1, 2]],
                       dtype='interval[int64, right]')
        """
        with rewrite_exception("IntervalArray", cls.__name__):
            arr = IntervalArray.from_tuples(data, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(arr, name=name)

    # --------------------------------------------------------------------
    # error: Return type "IntervalTree" of "_engine" incompatible with return type
    # "Union[IndexEngine, ExtensionEngine]" in supertype "Index"
    @cache_readonly
    def _engine(self) -> IntervalTree:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: cached interval-tree lookup engine. Converts the left and right bounds
        # to int64 representation when they are datetime-like and upcasts narrow numeric
        # types to 64-bit (the tree supports only 64-bit numeric data), then builds the
        # tree with the same closed side. Returns the tree. The signature intentionally
        # differs from the supertype.
        # IntervalTree does not supports numpy array unless they are 64 bit
        left = self._maybe_convert_i8(self.left)
        left = maybe_upcast_numeric_to_64bit(left)
        right = self._maybe_convert_i8(self.right)
        right = maybe_upcast_numeric_to_64bit(right)
        return IntervalTree(left, right, closed=self.closed)

    def __contains__(self, key: Any) -> bool:
        # [FUNCTION DOC]
        # Membership test. The key must be hashable. Only Interval objects can be
        # members: for other keys the answer is whether the key is a valid missing value
        # for the dtype and the index contains missing values, else false. For an
        # Interval, returns whether location lookup finds it (a key error means false).
        """
        return a boolean if this key is IN the index
        We *only* accept an Interval

        Parameters
        ----------
        key : Interval

        Returns
        -------
        bool
        """
        hash(key)
        if not isinstance(key, Interval):
            if is_valid_na_for_dtype(key, self.dtype):
                return self.hasnans
            return False

        try:
            self.get_loc(key)
            return True
        except KeyError:
            return False

    def _getitem_slice(self, slobj: slice) -> IntervalIndex:
        # [FUNCTION DOC]
        # HELPER: slice fast path. Parameter slobj is a slice; slices the array and wraps
        # the result in a new index of the same class keeping the name. Returns it.
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._data[slobj]
        return type(self)._simple_new(res, name=self._name)

    @cache_readonly
    def _multiindex(self) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: cached two-level MultiIndex whose levels are the left and right bounds,
        # named left and right.
        return MultiIndex.from_arrays([self.left, self.right], names=["left", "right"])

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns the reconstruction function with the class and a state
        # dictionary of the left bounds, right bounds, closed side and name, and no extra
        # state.
        d = {
            "left": self.left,
            "right": self.right,
            "closed": self.closed,
            "name": self.name,
        }
        return _new_IntervalIndex, (type(self), d), None

    @property
    def inferred_type(self) -> str:
        # [FUNCTION DOC]
        # Property: constant string "interval".
        """Return a string of the type inferred from the values"""
        return "interval"

    def memory_usage(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # Return the memory usage in bytes as the sum of the left and right bounds'
        # usage. Parameter deep (default false) is passed on to introspect object dtypes.
        # No engine memory is counted.
        """
        Memory usage of the values.

        Parameters
        ----------
        deep : bool, default False
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption.

        Returns
        -------
        bytes used
            Returns memory usage of the values in the Index in bytes.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False or if used on PyPy

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.memory_usage()
        24
        """
        # we don't use an explicit engine
        # so return the bytes here
        return self.left.memory_usage(deep=deep) + self.right.memory_usage(deep=deep)

    # IntervalTree doesn't have a is_monotonic_decreasing, so have to override
    #  the Index implementation
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: true if the index is monotonic decreasing, computed as whether
        # the reversed index is monotonic increasing. Overrides the base class because
        # the interval tree provides no decreasing check.
        """
        Return True if the IntervalIndex is monotonic decreasing (only equal or
        decreasing values), else False
        """
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def is_unique(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: true if all intervals are distinct. False if there is more than
        # one missing value; true if either bound array is unique on its own; otherwise
        # the pairs of bounds at positions where the left bound is duplicated are
        # collected in a set and a repeated pair gives false.
        """
        Return True if the IntervalIndex contains unique elements, else False.
        """
        left = self.left
        right = self.right

        if self.isna().sum() > 1:
            return False

        if left.is_unique or right.is_unique:
            return True

        seen_pairs = set()
        check_idx = np.where(left.duplicated(keep=False))[0]
        for idx in check_idx:
            pair = (left[idx], right[idx])
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)

        return True

    @property
    def is_overlapping(self) -> bool:
        # [FUNCTION DOC]
        # Property: true if any two intervals share a point (closed shared endpoints
        # count; open-only shared endpoints do not). Delegates to the interval tree
        # engine (GH 23309).
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Returns
        -------
        bool
            Boolean indicating if the IntervalIndex has overlapping intervals.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.
        IntervalIndex.overlaps : Check an IntervalIndex elementwise for
            overlaps.

        Examples
        --------
        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
        >>> index
        IntervalIndex([(0, 2], (1, 3], (4, 5]],
              dtype='interval[int64, right]')
        >>> index.is_overlapping
        True

        Intervals that share closed endpoints overlap:

        >>> index = pd.interval_range(0, 3, closed="both")
        >>> index
        IntervalIndex([[0, 1], [1, 2], [2, 3]],
              dtype='interval[int64, both]')
        >>> index.is_overlapping
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> index = pd.interval_range(0, 3, closed="left")
        >>> index
        IntervalIndex([[0, 1), [1, 2), [2, 3)],
              dtype='interval[int64, left]')
        >>> index.is_overlapping
        False
        """
        # GH 23309
        return self._engine.is_overlapping

    def _needs_i8_conversion(self, key) -> bool:
        # [FUNCTION DOC]
        # HELPER: tells whether a key must be converted to its integer representation
        # before querying the tree. Parameter key is a scalar or an index (list-likes
        # must have been cast already). True for timestamps, timedeltas and datetime or
        # timedelta indexes; for an Interval or an interval-dtype key the test is applied
        # recursively to its left bound. Returns a boolean.
        """
        Check if a given key needs i8 conversion. Conversion is necessary for
        Timestamp, Timedelta, DatetimeIndex, and TimedeltaIndex keys. An
        Interval-like requires conversion if its endpoints are one of the
        aforementioned types.

        Assumes that any list-like data has already been cast to an Index.

        Parameters
        ----------
        key : scalar or Index-like
            The key that should be checked for i8 conversion

        Returns
        -------
        bool
        """
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            return self._needs_i8_conversion(key.left)

        i8_types = (Timestamp, Timedelta, DatetimeIndex, TimedeltaIndex)
        return isinstance(key, i8_types)

    def _maybe_convert_i8(self, key):
        # [FUNCTION DOC]
        # HELPER: converts a key to the numeric form expected by the interval tree.
        # Parameter key is a scalar or list-like (list-likes are first cast to an Index
        # with narrow numerics upcast to 64-bit). If no conversion is needed the key is
        # returned unchanged. Intervals and interval indexes have both bounds converted
        # and are rebuilt (a scalar Interval or an IntervalIndex) with this index's closed
        # side. Scalars become their integer value (period ordinal, timestamp value, or
        # datetime/timedelta view); datetime-like indexes become int64 indexes with
        # missing values turned into not-a-number. Raises a value error if the key's
        # dtype does not match this index's subtype. Returns the converted key.
        """
        Maybe convert a given key to its equivalent i8 value(s). Used as a
        preprocessing step prior to IntervalTree queries (self._engine), which
        expects numeric data.

        Parameters
        ----------
        key : scalar or list-like
            The key that should maybe be converted to i8.

        Returns
        -------
        scalar or list-like
            The original key if no conversion occurred, int if converted scalar,
            Index with an int64 dtype if converted list-like.
        """
        if is_list_like(key):
            key = ensure_index(key)
            key = maybe_upcast_numeric_to_64bit(key)

        if not self._needs_i8_conversion(key):
            return key

        scalar = is_scalar(key)
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            # convert left/right and reconstruct
            left = self._maybe_convert_i8(key.left)
            right = self._maybe_convert_i8(key.right)
            constructor = Interval if scalar else IntervalIndex.from_arrays
            return constructor(left, right, closed=self.closed)

        if scalar:
            # Timestamp/Timedelta
            key_dtype, key_i8 = infer_dtype_from_scalar(key)
            if isinstance(key, Period):
                key_i8 = key.ordinal
            elif isinstance(key_i8, Timestamp):
                key_i8 = key_i8._value
            elif isinstance(key_i8, (np.datetime64, np.timedelta64)):
                key_i8 = key_i8.view("i8")
        else:
            # DatetimeIndex/TimedeltaIndex
            key_dtype, key_i8 = key.dtype, Index(key.asi8, copy=False)
            if key.hasnans:
                # convert NaT from its i8 value to np.nan so it's not viewed
                # as a valid value, maybe causing errors (e.g. is_overlapping)
                key_i8 = key_i8.where(~key._isnan)

        # ensure consistency with IntervalIndex subtype
        # error: Item "ExtensionDtype"/"dtype[Any]" of "Union[dtype[Any],
        # ExtensionDtype]" has no attribute "subtype"
        subtype = self.dtype.subtype  # type: ignore[union-attr]

        if subtype != key_dtype:
            raise ValueError(
                f"Cannot index an IntervalIndex of subtype {subtype} with "
                f"values of dtype {key_dtype}"
            )

        return key_i8

    def _searchsorted_monotonic(self, label, side: Literal["left", "right"] = "left"):
        # [FUNCTION DOC]
        # HELPER: binary search of a label in the monotonic structure. Parameters: label
        # (scalar) and side (left or right; default left). Raises a key error unless the
        # bounds are non-overlapping and monotonic, and a not-implemented error if the
        # label is an Interval or an IntervalIndex. Chooses the right or left bounds as
        # the search side depending on the side and the monotonic direction (GH 20921),
        # moving the label to the next or previous representable value when that bound is
        # open, and delegates to the bounds index's monotonic search. Returns the
        # insertion position.
        if not self.is_non_overlapping_monotonic:
            raise KeyError(
                "can only get slices from an IntervalIndex if bounds are "
                "non-overlapping and all monotonic increasing or decreasing"
            )

        if isinstance(label, (IntervalMixin, IntervalIndex)):
            raise NotImplementedError("Interval objects are not currently supported")

        # GH 20921: "not is_monotonic_increasing" for the second condition
        # instead of "is_monotonic_decreasing" to account for single element
        # indexes being both increasing and decreasing
        if (side == "left" and self.left.is_monotonic_increasing) or (
            side == "right" and not self.left.is_monotonic_increasing
        ):
            sub_idx = self.right
            if self.open_right:
                label = _get_next_label(label)
        else:
            sub_idx = self.left
            if self.open_left:
                label = _get_prev_label(label)

        return sub_idx._searchsorted_monotonic(label, side)

    # --------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key) -> int | slice | np.ndarray:
        # [FUNCTION DOC]
        # Return the location of a label. Parameter key is an Interval (exact match,
        # requires the same closed side, else key error), a missing value (matches the
        # missing entries), or a scalar point (matches all intervals containing it
        # according to the closed flags; a scalar not comparable to the subtype gives a
        # key error). Raises a key error when nothing matches (and from the indexing
        # check for non-scalar keys). Returns an integer for exactly one match, otherwise a
        # slice if the matches are contiguous (open-ended stops are completed to the
        # length), otherwise a boolean mask array.
        """
        Get integer location, slice or boolean mask for requested label.

        The `get_loc` method is used to retrieve the integer index, a slice for
        slicing objects, or a boolean mask indicating the presence of the label
        in the `IntervalIndex`.

        Parameters
        ----------
        key : label
            The value or range to find in the IntervalIndex.

        Returns
        -------
        int if unique index, slice if monotonic index, else mask
            The position or positions found. This could be a single
            number, a range, or an array of true/false values
            indicating the position(s) of the label.

        See Also
        --------
        IntervalIndex.get_indexer_non_unique : Compute indexer and
            mask for new index given the current index.
        Index.get_loc : Similar method in the base Index class.

        Examples
        --------
        >>> i1, i2 = pd.Interval(0, 1), pd.Interval(1, 2)
        >>> index = pd.IntervalIndex([i1, i2])
        >>> index.get_loc(1)
        0

        You can also supply a point inside an interval.

        >>> index.get_loc(1.5)
        1

        If a label is in several intervals, you get the locations of all the
        relevant intervals.

        >>> i3 = pd.Interval(0, 2)
        >>> overlapping_index = pd.IntervalIndex([i1, i2, i3])
        >>> overlapping_index.get_loc(0.5)
        array([ True, False,  True])

        Only exact matches will be returned if an interval is provided.

        >>> index.get_loc(pd.Interval(0, 1))
        0
        """
        self._check_indexing_error(key)

        if isinstance(key, Interval):
            if self.closed != key.closed:
                raise KeyError(key)
            mask = (self.left == key.left) & (self.right == key.right)
        elif is_valid_na_for_dtype(key, self.dtype):
            mask = self.isna()
        else:
            # assume scalar
            op_left = le if self.closed_left else lt
            op_right = le if self.closed_right else lt
            try:
                mask = op_left(self.left, key) & op_right(key, self.right)
            except TypeError as err:
                # scalar is not comparable to II subtype --> invalid label
                raise KeyError(key) from err

        matches = mask.sum()
        if matches == 0:
            raise KeyError(key)
        if matches == 1:
            return maybe_unbox_numpy_scalar(mask.argmax())

        res = lib.maybe_booleans_to_slice(mask.view("u1"))
        if isinstance(res, slice) and res.stop is None:
            # TODO: DO this in maybe_booleans_to_slice?
            res = slice(res.start, len(self), res.step)
        return res

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance: Any | None = None,
    ) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: positions of target labels in this index (index must be non-overlapping
        # when called). Parameters: target; method, limit, tolerance are unused. Interval
        # targets: if both sides of this index are unique, positions are the entries where
        # the left and right lookups agree; otherwise the pointwise method is used.
        # Targets of non-object, non-string dtype use the interval tree (after
        # integer conversion). Object or string targets use the pointwise method.
        # Returns a platform integer array with minus one for missing.
        if isinstance(target, IntervalIndex):
            # We only get here with not self.is_overlapping
            # -> at most one match per interval in target
            # want exact matches -> need both left/right to match, so defer to
            # left/right get_indexer, compare elementwise, equality -> match
            if self.left.is_unique and self.right.is_unique:
                indexer = self._get_indexer_unique_sides(target)
            else:
                indexer = self._get_indexer_pointwise(target)[0]

        elif not (is_object_dtype(target.dtype) or is_string_dtype(target.dtype)):
            # homogeneous scalar index: use IntervalTree
            # we should always have self._should_partial_index(target) here
            target = self._maybe_convert_i8(target)
            indexer = self._engine.get_indexer(target.values)
        else:
            # heterogeneous scalar index: defer elementwise to get_loc
            # we should always have self._should_partial_index(target) here
            return self._get_indexer_pointwise(target)[0]

        return ensure_platform_int(indexer)

    def get_indexer_non_unique(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # Compute positions in this index for each target element, allowing duplicates.
        # Parameter target is an index or list of intervals. Returns a pair: the indexer
        # (positions, with minus one for targets not found) and the array of target
        # positions that were missing. If the dtypes cannot be compared and cannot be
        # partially indexed everything is missing. Interval targets use the fast path
        # when both sides are unique, else the pointwise method. Object targets or
        # targets that cannot be partially indexed use the pointwise method. Other
        # scalar targets go through the tree engine.
        # KNOWN ISSUE: the docstring examples use a plain Index of strings and were evidently copied from the base class; they do not exercise an IntervalIndex.
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : IntervalIndex or list of Intervals
            An iterable containing the values to be used for computing indexer.

        Returns
        -------
        indexer : np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.
        missing : np.ndarray[np.intp]
            An indexer into the target of the values not found.
            These correspond to the -1 in the indexer array.

        See Also
        --------
        Index.get_indexer : Computes indexer and mask for new index given
            the current index.
        Index.get_indexer_for : Returns an indexer even when non-unique.

        Examples
        --------
        >>> index = pd.Index(["c", "b", "a", "b", "b"])
        >>> index.get_indexer_non_unique(["b", "b"])
        (array([1, 3, 4, 1, 3, 4]), array([], dtype=int64))

        In the example below there are no matched values.

        >>> index = pd.Index(["c", "b", "a", "b", "b"])
        >>> index.get_indexer_non_unique(["q", "r", "t"])
        (array([-1, -1, -1]), array([0, 1, 2]))

        For this reason, the returned ``indexer`` contains only integers equal to -1.
        It demonstrates that there's no match between the index and the ``target``
        values at these positions. The mask [0, 1, 2] in the return value shows that
        the first, second, and third elements are missing.

        Notice that the return value is a tuple contains two items. In the example
        below the first item is an array of locations in ``index``. The second
        item is a mask shows that the first and third elements are missing.

        >>> index = pd.Index(["c", "b", "a", "b", "b"])
        >>> index.get_indexer_non_unique(["f", "b", "s"])
        (array([-1,  1,  3,  4, -1]), array([0, 2]))
        """
        target = ensure_index(target)

        if not self._should_compare(target) and not self._should_partial_index(target):
            # e.g. IntervalIndex with different closed or incompatible subtype
            #  -> no matches
            return self._get_indexer_non_comparable(target, None, unique=False)

        elif isinstance(target, IntervalIndex):
            if self.left.is_unique and self.right.is_unique:
                # fastpath available even if we don't have self._index_as_unique
                indexer = self._get_indexer_unique_sides(target)
                missing = (indexer == -1).nonzero()[0]
            else:
                return self._get_indexer_pointwise(target)

        elif is_object_dtype(target.dtype) or not self._should_partial_index(target):
            # target might contain intervals: defer elementwise to get_loc
            return self._get_indexer_pointwise(target)

        else:
            # Note: this case behaves differently from other Index subclasses
            #  because IntervalIndex does partial-int indexing
            target = self._maybe_convert_i8(target)
            indexer, missing = self._engine.get_indexer_non_unique(target.values)

        return ensure_platform_int(indexer), ensure_platform_int(missing)

    def _get_indexer_unique_sides(self, target: IntervalIndex) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: exact-interval lookup when both bounds of this index are unique. Parameter
        # target is an IntervalIndex. The left bounds and right bounds are looked up
        # separately and a position counts only when both lookups agree; else minus one.
        # The caller must have checked uniqueness of both sides. Returns the position
        # array.
        """
        _get_indexer specialized to the case where both of our sides are unique.
        """
        # Caller is responsible for checking
        #  `self.left.is_unique and self.right.is_unique`

        left_indexer = self.left.get_indexer(target.left)
        right_indexer = self.right.get_indexer(target.right)
        indexer = np.where(left_indexer == right_indexer, left_indexer, -1)
        return indexer

    def _get_indexer_pointwise(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # HELPER: elementwise lookup used for get_indexer and its non-unique variant.
        # Parameter target is an index. For each element, calls location lookup: slices
        # are expanded to positions, an integer is wrapped, and a mask is converted to
        # positions; a key error or an invalid-index error (non-scalar key such as a
        # tuple) records the target position as missing and contributes minus one.
        # Returns a pair of platform integer arrays: concatenated positions and missing
        # target positions.
        """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
        indexer, missing = [], []
        for i, key in enumerate(target):
            try:
                locs = self.get_loc(key)
                if isinstance(locs, slice):
                    # Only needed for get_indexer_non_unique
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype="intp")
                elif lib.is_integer(locs):
                    locs = np.array(locs, ndmin=1)
                else:
                    # otherwise we have ndarray[bool]
                    locs = np.where(locs)[0]
            except KeyError:
                missing.append(i)
                locs = np.array([-1])
            except InvalidIndexError:
                # i.e. non-scalar key e.g. a tuple.
                # see test_append_different_columns_types_raises
                missing.append(i)
                locs = np.array([-1])

            indexer.append(locs)

        concatenated_indexer = np.concatenate(indexer)
        return ensure_platform_int(concatenated_indexer), ensure_platform_int(missing)

    @cache_readonly
    def _index_as_unique(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached flag used by the indexer machinery: true when the intervals do
        # not overlap and there is at most one missing value.
        return not self.is_overlapping and self._engine._na_count < 2

    _requires_unique_msg = (
        "cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique"
    )

    def _convert_slice_indexer(self, key: slice, kind: Literal["loc", "getitem"]):
        # [FUNCTION DOC]
        # HELPER: validates slice indexers. Parameters: key (a slice) and kind (loc or
        # getitem). A step other than none or one raises a value error for label-based
        # (loc) slicing, and for getitem unless the slice can be interpreted as
        # positional (GH#31658). Otherwise defers to the base class and returns its
        # result.
        if not (key.step is None or key.step == 1):
            # GH#31658 if label-based, we require step == 1,
            #  if positional, we disallow float start/stop
            msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
            if kind == "loc":
                raise ValueError(msg)
            if kind == "getitem":
                if not is_valid_positional_slice(key):
                    # i.e. this cannot be interpreted as a positional slice
                    raise ValueError(msg)

        return super()._convert_slice_indexer(key, kind)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached property saying whether integer lookups are positional; true
        # only when the interval subtype is datetime-like or timedelta-like.
        # integer lookups in Series.__getitem__ are unambiguously
        #  positional in this case
        # error: Item "ExtensionDtype"/"dtype[Any]" of "Union[dtype[Any],
        # ExtensionDtype]" has no attribute "subtype"
        return self.dtype.subtype.kind in "mM"  # type: ignore[union-attr]

    def _maybe_cast_slice_bound(self, label, side: str):
        # [FUNCTION DOC]
        # HELPER: slice-bound casting hook. Parameters: label and side (left or right).
        # Delegates to the same-named bounds index (left bounds for side left, right
        # bounds for side right) and returns its result.
        return getattr(self, side)._maybe_cast_slice_bound(label, side)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        # [FUNCTION DOC]
        # HELPER: comparability check. False unless the dtype is an interval dtype; true
        # if the common type of the two dtypes is not object.
        if not isinstance(dtype, IntervalDtype):
            return False
        common_subtype = find_common_type([self.dtype, dtype])
        return not is_object_dtype(common_subtype)

    # --------------------------------------------------------------------

    @cache_readonly
    def left(self) -> Index:
        # [FUNCTION DOC]
        # Cached property: the left bounds as an Index (no copy).
        """
        Return left bounds of the intervals in the IntervalIndex.

        The left bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the left bounds is the
        same as the datatype of the endpoints of the intervals.

        Returns
        -------
        Index
            An Index containing the left bounds of the intervals.

        See Also
        --------
        IntervalIndex.right : Return the right bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.mid : Return the mid-point of the intervals in
            the IntervalIndex.
        IntervalIndex.length : Return the length of the intervals in
            the IntervalIndex.

        Examples
        --------
        >>> iv_idx = pd.IntervalIndex.from_arrays([1, 2, 3], [4, 5, 6], closed="right")
        >>> iv_idx.left
        Index([1, 2, 3], dtype='int64')

        >>> iv_idx = pd.IntervalIndex.from_tuples(
        ...     [(1, 4), (2, 5), (3, 6)], closed="left"
        ... )
        >>> iv_idx.left
        Index([1, 2, 3], dtype='int64')
        """
        return Index(self._data.left, copy=False)

    @cache_readonly
    def right(self) -> Index:
        # [FUNCTION DOC]
        # Cached property: the right bounds as an Index (no copy).
        """
        Return right bounds of the intervals in the IntervalIndex.

        The right bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the right bounds is the
        same as the datatype of the endpoints of the intervals.

        Returns
        -------
        Index
            An Index containing the right bounds of the intervals.

        See Also
        --------
        IntervalIndex.left : Return the left bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.mid : Return the mid-point of the intervals in
            the IntervalIndex.
        IntervalIndex.length : Return the length of the intervals in
            the IntervalIndex.

        Examples
        --------
        >>> iv_idx = pd.IntervalIndex.from_arrays([1, 2, 3], [4, 5, 6], closed="right")
        >>> iv_idx.right
        Index([4, 5, 6], dtype='int64')

        >>> iv_idx = pd.IntervalIndex.from_tuples(
        ...     [(1, 4), (2, 5), (3, 6)], closed="left"
        ... )
        >>> iv_idx.right
        Index([4, 5, 6], dtype='int64')
        """
        return Index(self._data.right, copy=False)

    @cache_readonly
    def mid(self) -> Index:
        # [FUNCTION DOC]
        # Cached property: the midpoints of the intervals as an Index (no copy); the
        # average of the bounds regardless of closedness.
        """
        Return the midpoint of each interval in the IntervalIndex as an Index.

        Each midpoint is calculated as the average of the left and right bounds
        of each interval. The midpoints are returned as a pandas Index object.

        Returns
        -------
        pandas.Index
            An Index containing the midpoints of each interval.

        See Also
        --------
        IntervalIndex.left : Return the left bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.right : Return the right bounds of the intervals
            in the IntervalIndex.
        IntervalIndex.length : Return the length of the intervals in
            the IntervalIndex.

        Notes
        -----
        The midpoint is the average of the interval bounds, potentially resulting
        in a floating-point number even if bounds are integers. The returned Index
        will have a dtype that accurately holds the midpoints. This computation is
        the same regardless of whether intervals are open or closed.

        Examples
        --------
        >>> iv_idx = pd.IntervalIndex.from_arrays([1, 2, 3], [4, 5, 6])
        >>> iv_idx.mid
        Index([2.5, 3.5, 4.5], dtype='float64')

        >>> iv_idx = pd.IntervalIndex.from_tuples([(1, 4), (2, 5), (3, 6)])
        >>> iv_idx.mid
        Index([2.5, 3.5, 4.5], dtype='float64')
        """
        return Index(self._data.mid, copy=False)

    @property
    def length(self) -> Index:
        # [FUNCTION DOC]
        # Property: the lengths of the intervals (right minus left) as an Index (no copy).
        """
        Calculate the length of each interval in the IntervalIndex.

        This method returns a new Index containing the lengths of each interval
        in the IntervalIndex. The length of an interval is defined as the difference
        between its end and its start.

        Returns
        -------
        Index
            An Index containing the lengths of each interval.

        See Also
        --------
        Interval.length : Return the length of the Interval.

        Examples
        --------
        >>> intervals = pd.IntervalIndex.from_arrays(
        ...     [1, 2, 3], [4, 5, 6], closed="right"
        ... )
        >>> intervals.length
        Index([3, 3, 3], dtype='int64')

        >>> intervals = pd.IntervalIndex.from_tuples([(1, 5), (6, 10), (11, 15)])
        >>> intervals.length
        Index([4, 4, 4], dtype='int64')
        """
        return Index(self._data.length, copy=False)

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other, sort: bool = False):
        # [FUNCTION DOC]
        # HELPER: intersection for matching dtypes (closed sides already equal).
        # Parameters: other and sort (default false). If this index has unique sides, the
        # unique-sides routine is used; otherwise if other has unique sides and this
        # index has at most one missing value, the roles are swapped; else the
        # non-unique routine runs. If sort is true the result is sorted. Returns the
        # intersection.
        """
        intersection specialized to the case with matching dtypes.
        """
        # For IntervalIndex we also know other.closed == self.closed
        if self.left.is_unique and self.right.is_unique:
            taken = self._intersection_unique(other)
        elif other.left.is_unique and other.right.is_unique and self.isna().sum() <= 1:
            # Swap other/self if other is unique and self does not have
            # multiple NaNs
            taken = other._intersection_unique(self)
        else:
            # duplicates
            taken = self._intersection_non_unique(other)

        if sort:
            taken = taken.sort_values()

        return taken

    def _intersection_unique(self, other: IntervalIndex) -> IntervalIndex:
        # [FUNCTION DOC]
        # HELPER: intersection when the endpoints are unique. Parameter other is an
        # IntervalIndex. Looks up the bounds of other in this index's left and right
        # bounds and keeps positions where both agree and exist; duplicates are removed
        # and the corresponding elements of this index are taken. Returns the result
        # index.
        """
        Used when the IntervalIndex does not have any common endpoint,
        no matter left or right.
        Return the intersection with another IntervalIndex.
        Parameters
        ----------
        other : IntervalIndex
        Returns
        -------
        IntervalIndex
        """
        # Note: this is much more performant than super()._intersection(other)
        lindexer = self.left.get_indexer(other.left)
        rindexer = self.right.get_indexer(other.right)

        match = (lindexer == rindexer) & (lindexer != -1)
        indexer = lindexer.take(match.nonzero()[0])
        indexer = unique(indexer)

        return self.take(indexer)

    def _intersection_non_unique(self, other: IntervalIndex) -> IntervalIndex:
        # [FUNCTION DOC]
        # HELPER: intersection when endpoints may repeat. Parameter other is an
        # IntervalIndex. Marks positions of this index whose (left, right) pair appears in
        # other (compared as a set of pairs, which requires both sides to be same
        # length), plus the first missing position when both have missing values.
        # Returns this index filtered by that mask, so duplicates in this index are kept.
        """
        Used when the IntervalIndex does have some common endpoints,
        on either sides.
        Return the intersection with another IntervalIndex.

        Parameters
        ----------
        other : IntervalIndex

        Returns
        -------
        IntervalIndex
        """
        # Note: this is about 3.25x faster than super()._intersection(other)
        #  in IntervalIndexMethod.time_intersection_both_duplicate(1000)
        mask = np.zeros(len(self), dtype=bool)

        if self.hasnans and other.hasnans:
            first_nan_loc = np.arange(len(self))[self.isna()][0]
            mask[first_nan_loc] = True

        other_tups = set(zip(other.left, other.right, strict=True))
        for i, tup in enumerate(zip(self.left, self.right, strict=True)):
            if tup in other_tups:
                mask[i] = True

        return self[mask]

    # --------------------------------------------------------------------

    def _get_engine_target(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: always raises a not-implemented error: this index does not pass values
        # to the numeric engines or the join fast paths.
        # Note: we _could_ use libjoin functions by either casting to object
        #  dtype or constructing tuples (faster than constructing Intervals)
        #  but the libjoin fastpaths are no longer fast in these cases.
        raise NotImplementedError(
            "IntervalIndex does not use libjoin fastpaths or pass values to "
            "IndexEngine objects"
        )

    def _from_join_target(self, result):
        # [FUNCTION DOC]
        # HELPER: always raises a not-implemented error: the join fast paths are unused.
        raise NotImplementedError("IntervalIndex does not use libjoin fastpaths")

    # TODO: arithmetic operations


def _is_valid_endpoint(endpoint) -> bool:
    # [FUNCTION DOC]
    # HELPER: validity check of start/end for the interval range builder. Returns true
    # if the endpoint is a number, a Timestamp, a Timedelta or none.
    """
    Helper for interval_range to check if start/end are valid types.
    """
    return any(
        [
            is_number(endpoint),
            isinstance(endpoint, Timestamp),
            isinstance(endpoint, Timedelta),
            endpoint is None,
        ]
    )


def _is_type_compatible(a, b) -> bool:
    # [FUNCTION DOC]
    # HELPER: type-compatibility check for the interval range builder. Parameters a
    # and b are start, end or freq values. Returns true if both are numbers, both are
    # timestamp-or-offset objects, both are timedelta-or-offset objects, or at least
    # one is none. The two predicates it uses are anonymous inline functions.
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
    is_ts_compat = lambda x: isinstance(x, (Timestamp, BaseOffset))
    is_td_compat = lambda x: isinstance(x, (Timedelta, BaseOffset))
    return (
        (is_number(a) and is_number(b))
        or (is_ts_compat(a) and is_ts_compat(b))
        or (is_td_compat(a) and is_td_compat(b))
        or com.any_none(a, b)
    )


@set_module("pandas")
def interval_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name: Hashable | None = None,
    closed: IntervalClosedType = "right",
) -> IntervalIndex:
    # [FUNCTION DOC]
    # Build an IntervalIndex with regular spacing. Parameters: start, end (numeric or
    # datetime-like; default none); periods (number of intervals; default none);
    # freq (number, string, Timedelta, timedelta or offset; default none, which
    # becomes one for numeric and one day otherwise when any of start, end, periods
    # is missing); name (default none); closed (default right).
    # Raises a value error unless exactly three of start, end, periods, freq are
    # given, for invalid start/end types, and for a freq that is neither numeric nor
    # convertible to an offset; raises a type error for incompatible start/end/freq
    # types. Periods is validated and increased by one to get the number of break
    # points. Numeric endpoints: when start, end and freq are all given the breaks
    # come from an arithmetic range extended slightly so the end is included, with
    # int64 or float64 dtype (or the shared dtype of numpy scalars); otherwise the
    # missing one of periods, start, end is derived and the breaks are linearly
    # spaced; integer inputs downcast the result to integers. Timestamp endpoints use
    # the date range builder, otherwise the timedelta range builder. Returns
    # the index built from the breaks with the given name, closed side and a matching
    # interval dtype.
    """
    Return a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric or datetime-like, default None
        Left bound for generating intervals.
    end : numeric or datetime-like, default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate.
    freq : numeric, str, Timedelta, datetime.timedelta, or DateOffset, default None
        The length of each interval. Must be consistent with the type of start
        and end, e.g. 2 for numeric, or '5H' for datetime-like.  Default is 1
        for numeric and 'D' for datetime-like.
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {'left', 'right', 'both', 'neither'}, default 'right'
        Whether the intervals are closed on the left-side, right-side, both
        or neither.

    Returns
    -------
    IntervalIndex
        Object with a fixed frequency.

    See Also
    --------
    IntervalIndex : An Index of intervals that are all closed on the same side.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``IntervalIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end``, inclusively.

    To learn more about datetime-like frequency strings, please see
    :ref:`this link<timeseries.offset_aliases>`.

    Examples
    --------
    Numeric ``start`` and  ``end`` is supported.

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    Additionally, datetime-like input is also supported.

    >>> pd.interval_range(
    ...     start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2017-01-04")
    ... )
    IntervalIndex([(2017-01-01 00:00:00, 2017-01-02 00:00:00],
                   (2017-01-02 00:00:00, 2017-01-03 00:00:00],
                   (2017-01-03 00:00:00, 2017-01-04 00:00:00]],
                  dtype='interval[datetime64[us], right]')

    The ``freq`` parameter specifies the frequency between the left and right.
    endpoints of the individual intervals within the ``IntervalIndex``.  For
    numeric ``start`` and ``end``, the frequency must also be numeric.

    >>> pd.interval_range(start=0, periods=4, freq=1.5)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
                  dtype='interval[float64, right]')

    Similarly, for datetime-like ``start`` and ``end``, the frequency must be
    convertible to a DateOffset.

    >>> pd.interval_range(start=pd.Timestamp("2017-01-01"), periods=3, freq="MS")
    IntervalIndex([(2017-01-01 00:00:00, 2017-02-01 00:00:00],
                   (2017-02-01 00:00:00, 2017-03-01 00:00:00],
                   (2017-03-01 00:00:00, 2017-04-01 00:00:00]],
                  dtype='interval[datetime64[us], right]')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.interval_range(start=0, end=6, periods=4)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
              dtype='interval[float64, right]')

    The ``closed`` parameter specifies which endpoints of the individual
    intervals within the ``IntervalIndex`` are closed.

    >>> pd.interval_range(end=5, periods=4, closed="both")
    IntervalIndex([[1, 2], [2, 3], [3, 4], [4, 5]],
                  dtype='interval[int64, both]')
    """
    start = maybe_box_datetimelike(start)
    end = maybe_box_datetimelike(end)
    endpoint = start if start is not None else end

    if freq is None and com.any_none(periods, start, end):
        freq = 1 if is_number(endpoint) else "D"

    if com.count_not_none(start, end, periods, freq) != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

    if not _is_valid_endpoint(start):
        raise ValueError(f"start must be numeric or datetime-like, got {start}")
    if not _is_valid_endpoint(end):
        raise ValueError(f"end must be numeric or datetime-like, got {end}")

    periods = validate_periods(periods)

    if freq is not None and not is_number(freq):
        try:
            freq = to_offset(freq)
        except ValueError as err:
            raise ValueError(
                f"freq must be numeric or convertible to DateOffset, got {freq}"
            ) from err

    # verify type compatibility
    if not all(
        [
            _is_type_compatible(start, end),
            _is_type_compatible(start, freq),
            _is_type_compatible(end, freq),
        ]
    ):
        raise TypeError("start, end, freq need to be type compatible")

    # +1 to convert interval count to breaks count (n breaks = n-1 intervals)
    if periods is not None:
        periods += 1

    breaks: np.ndarray | TimedeltaIndex | DatetimeIndex

    if is_number(endpoint):
        dtype: np.dtype = np.dtype("int64")
        if com.all_not_none(start, end, freq):
            if (
                isinstance(start, (np.integer, np.floating))
                and isinstance(end, (np.integer, np.floating))
                and start.dtype == end.dtype
            ):
                dtype = start.dtype
            elif (
                isinstance(start, (float, np.floating))
                or isinstance(end, (float, np.floating))
                or isinstance(freq, (float, np.floating))
            ):
                dtype = np.dtype("float64")
            # 0.1 ensures we capture end
            breaks = np.arange(start, end + (freq * 0.1), freq)
            breaks = maybe_downcast_numeric(breaks, dtype)
        else:
            # compute the period/start/end if unspecified (at most one)
            if periods is None:
                periods = int((end - start) // freq) + 1
            elif start is None:
                start = end - (periods - 1) * freq
            elif end is None:
                end = start + (periods - 1) * freq

            breaks = np.linspace(start, end, periods)
        if all(is_integer(x) for x in com.not_none(start, end, freq)):
            # np.linspace always produces float output
            breaks = maybe_downcast_numeric(breaks, dtype)
    # delegate to the appropriate range function
    elif isinstance(endpoint, Timestamp):
        breaks = date_range(start=start, end=end, periods=periods, freq=freq)
    else:
        breaks = timedelta_range(start=start, end=end, periods=periods, freq=freq)

    return IntervalIndex.from_breaks(
        breaks,
        name=name,
        closed=closed,
        dtype=IntervalDtype(subtype=breaks.dtype, closed=closed),
    )
