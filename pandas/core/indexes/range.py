# [MODULE DOC]
# Implementation of RangeIndex, a memory-saving index representing a monotonic
# arithmetic progression of 64-bit integers, stored as a built-in range object.
# Public contents: the RangeIndex class (re-labelled to the top-level package
# namespace) and a small module-level helper returning the smallest element of a
# progression that is at least a limit. Module constants: an empty range and the
# int64 dtype object.
# Relationships: subclasses the generic Index. Many operations are overridden to
# return a RangeIndex when the result is still an arithmetic progression
# (slicing, set operations, insert, delete, negation, arithmetic with integers,
# concatenation, taking, sorting) and otherwise fall back to the base class,
# which materializes an int64 Index. The integer array form is only built lazily
# and cached. Imports nothing at runtime beyond the base index machinery; the
# series class is imported lazily inside one method. No import-time side effects.

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterator,
)
from datetime import timedelta
import operator
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.internals import BlockValuesRefs
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    ensure_platform_int,
    ensure_python_int,
    is_float,
    is_integer,
    is_scalar,
    is_signed_integer_dtype,
)
from pandas.core.dtypes.generic import ABCTimedeltaIndex

from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        Dtype,
        JoinHow,
        NaPosition,
        NumpySorter,
        NumpyValueArrayLike,
        ScalarLike_co,
        npt,
    )

    from pandas import Series
    from pandas.core.arrays import ExtensionArray

_empty_range = range(0)
_dtype_int64 = np.dtype(np.int64)


def min_fitting_element(start: int, step: int, lower_limit: int) -> int:
    # [FUNCTION DOC]
    # HELPER: smallest value of the form start plus a multiple of the absolute step
    # that is greater than or equal to lower_limit. Parameters: start, step (sign is
    # ignored; must be nonzero or a zero-division error occurs) and lower_limit, all
    # integers. Returns that integer. Used by the intersection logic.
    """Returns the smallest element greater than or equal to the limit"""
    no_steps = -(-(lower_limit - start) // abs(step))
    return start + abs(step) * no_steps


@set_module("pandas")
class RangeIndex(Index):
    # [CLASS DOC]
    # Immutable index over a Python range object, representing an arithmetic
    # progression of int64 values. Inherits from Index, which supplies everything not
    # overridden. State: _range is the immutable range (start, stop and step are read
    # from it); _name is the name; _cache is a per-instance cache dictionary in
    # which the materialized integer array is stored lazily. The dtype is always
    # int64 and the index is always unique and monotonic (increasing for a positive
    # step, decreasing for a negative step). Invariants: step is never zero; the
    # instance is built only through the private simple constructor (which asserts a
    # range object). Default index of series and data frames. Key behaviors:
    # operations that keep the result a progression return a RangeIndex, others
    # return a plain integer Index.
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int, range, or other RangeIndex instance, default None
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int, default None
        The end value of the range (exclusive).
    step : int, default None
        The step size of the range.
    dtype : np.int64, default None
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []

    >>> list(pd.RangeIndex(1, 0))
    []
    """

    _typ = "rangeindex"
    _dtype_validation_metadata = (is_signed_integer_dtype, "signed integer")
    _range: range
    _values: np.ndarray

    @property
    def _engine_type(self) -> type[libindex.Int64Engine]:
        # [FUNCTION DOC]
        # HELPER: returns the Cython hash-table engine class for lookups, the int64
        # engine (used only when the base class machinery needs an engine).
        return libindex.Int64Engine

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        start=None,
        stop=None,
        step=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Construct a RangeIndex. Parameters mirror the builtin range: start, stop, step
        # (integers; default none); dtype (validated but otherwise unused; must be a
        # signed integer dtype if given); copy (unused); name (default none, taken from
        # start if it has a name). Returns a RangeIndex. Raises a value error for an
        # invalid dtype or a zero step, and a type error when called with no arguments
        # at all or with non-integer values (from the integer coercion helper).
        # Behavior: if start is already a RangeIndex it is copied with the resolved name;
        # if start is a range object it is used directly; otherwise start defaults to
        # zero, a lone argument is treated as the stop with start zero, and step
        # defaults to one.
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)

        # RangeIndex
        if isinstance(start, cls):
            return start.copy(name=name)
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)

        # validate the arguments
        if com.all_none(start, stop, step):
            raise TypeError("RangeIndex(...) must be called with integers")

        start = ensure_python_int(start) if start is not None else 0

        if stop is None:
            start, stop = 0, start
        else:
            stop = ensure_python_int(stop)

        step = ensure_python_int(step) if step is not None else 1
        if step == 0:
            raise ValueError("Step must not be zero")

        rng = range(start, stop, step)
        return cls._simple_new(rng, name=name)

    @classmethod
    def from_range(cls, data: range, name=None, dtype: Dtype | None = None) -> Self:
        # [FUNCTION DOC]
        # Class method creating a RangeIndex from a builtin range object. Parameters:
        # data must be a range; name (default none); dtype (default none), validated as
        # in the constructor and otherwise unused. Raises a type error if data is not a
        # range, and a value error for an invalid dtype. Returns the new index.
        """
        Create :class:`pandas.RangeIndex` from a ``range`` object.

        This method provides a way to create a :class:`pandas.RangeIndex` directly
        from a Python ``range`` object. The resulting :class:`RangeIndex` will have
        the same start, stop, and step values as the input ``range`` object.
        It is particularly useful for constructing indices in an efficient and
        memory-friendly manner.

        Parameters
        ----------
        data : range
            The range object to be converted into a RangeIndex.
        name : str, default None
            Name to be stored in the index.
        dtype : Dtype or None
            Data type for the RangeIndex. If None, the default integer type will
            be used.

        Returns
        -------
        RangeIndex

        See Also
        --------
        RangeIndex : Immutable Index implementing a monotonic integer range.
        Index : Immutable sequence used for indexing and alignment.

        Examples
        --------
        >>> pd.RangeIndex.from_range(range(5))
        RangeIndex(start=0, stop=5, step=1)

        >>> pd.RangeIndex.from_range(range(2, -10, -3))
        RangeIndex(start=2, stop=-10, step=-3)
        """
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a "
                f"range, {data!r} was passed"
            )
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    #  error: Argument 1 of "_simple_new" is incompatible with supertype "Index";
    #  supertype defines the argument type as
    #  "Union[ExtensionArray, ndarray[Any, Any]]"  [override]
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls, values: range, name: Hashable | None = None
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: private fast constructor. Parameters: values must be a range
        # (asserted); name (default none). Creates an instance without calling the
        # normal constructor, sets the range, name and an empty cache, resets the
        # identity token, and leaves the reference tracker to be created lazily.
        # Returns the instance.
        result = object.__new__(cls)

        assert isinstance(values, range)

        result._range = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        # result._references populated lazily
        return result

    @cache_readonly
    def _references(self) -> BlockValuesRefs:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: cached tracker of shared-data references used for copy-on-write,
        # created on first use and registering this index with itself.
        result = BlockValuesRefs()
        result.add_index_reference(self)
        return result

    @classmethod
    def _validate_dtype(cls, dtype: Dtype | None) -> None:
        # [FUNCTION DOC]
        # HELPER: class method checking a dtype argument. Parameter dtype may be none
        # (accepted). Otherwise it is tested with the class's validation function
        # (signed integer check); raises a value error naming the expected kind and the
        # received dtype on failure. Returns nothing.
        if dtype is None:
            return

        validation_func, expected = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(
                f"Incorrect `dtype` passed: expected {expected}, received {dtype}"
            )

    # --------------------------------------------------------------------

    # error: Return type "Type[Index]" of "_constructor" incompatible with return
    # type "Type[RangeIndex]" in supertype "Index"
    @cache_readonly
    def _constructor(self) -> type[Index]:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: cached class to use when constructing results of operations on this
        # index; always the generic Index class (not RangeIndex), since results are not
        # generally progressions. The signature intentionally differs from the
        # supertype.
        """return the class to use for construction"""
        return Index

    # error: Signature of "_data" incompatible with supertype "Index"
    @cache_readonly
    def _data(self) -> np.ndarray:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: cached int64 numpy array of all values, materialized on demand from
        # the range and stored in the instance cache.
        """
        An int array that for performance reasons is created only when needed.

        The constructed array is saved in ``_cache``.
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def _get_data_as_items(self) -> list[tuple[str, int]]:
        # [FUNCTION DOC]
        # HELPER: returns a list of three pairs naming and giving the start, stop and
        # step of the underlying range, used for printing and pickling.
        """return a list of tuples of start, stop, step"""
        rng = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns the shared base-index reconstruction function with the
        # class and a state dictionary containing the name, start, stop and step, and no
        # extra state.
        d = {"name": self._name}
        d.update(dict(self._get_data_as_items()))
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------
    # Rendering Methods

    def _format_attrs(self):
        # [FUNCTION DOC]
        # HELPER: returns the list of (attribute, value) pairs shown in the
        # representation: start, stop and step, plus the name (pretty-printed) when it
        # is set.
        """
        Return a list of tuples of the (attr, formatted_value)
        """
        attrs = cast("list[tuple[str, str | int]]", self._get_data_as_items())
        if self._name is not None:
            attrs.append(("name", ibase.default_pprint(self._name)))
        return attrs

    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: formats the values as strings for display, with a header prepended.
        # Parameters: header is the list of header strings; na_rep is the text for
        # missing values (unused, since a range has no missing values). Returns the
        # header alone when empty; otherwise the header followed by every value converted
        # to text and left-justified to the width of the longer of the first and last
        # values. Faster equivalent of the base implementation.
        # Equivalent to Index implementation, but faster
        if not len(self._range):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[-1])
        max_length = max(len(first_val_str), len(last_val_str))

        return header + [f"{x:<{max_length}}" for x in self._range]

    # --------------------------------------------------------------------

    @property
    def start(self) -> int:
        # [FUNCTION DOC]
        # Property: the start value of the range (zero when not supplied at creation).
        """
        The value of the `start` parameter (``0`` if this was not supplied).

        This property returns the starting value of the `RangeIndex`. If the `start`
        value is not explicitly provided during the creation of the `RangeIndex`,
        it defaults to 0.

        See Also
        --------
        RangeIndex : Immutable index implementing a range-based index.
        RangeIndex.stop : Returns the stop value of the `RangeIndex`.
        RangeIndex.step : Returns the step value of the `RangeIndex`.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.start
        0

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.start
        2
        """
        # GH 25710
        return self._range.start

    @property
    def stop(self) -> int:
        # [FUNCTION DOC]
        # Property: the exclusive stop value of the range.
        """
        The value of the `stop` parameter.

        This property returns the `stop` value of the RangeIndex, which defines the
        upper (or lower, in case of negative steps) bound of the index range. The
        `stop` value is exclusive, meaning the RangeIndex includes values up to but
        not including this value.

        See Also
        --------
        RangeIndex : Immutable index representing a range of integers.
        RangeIndex.start : The start value of the RangeIndex.
        RangeIndex.step : The step size between elements in the RangeIndex.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.stop
        5

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.stop
        -10
        """
        return self._range.stop

    @property
    def step(self) -> int:
        # [FUNCTION DOC]
        # Property: the step of the range (one when not supplied at creation), even for
        # an empty range.
        """
        The value of the `step` parameter (``1`` if this was not supplied).

        The ``step`` parameter determines the increment (or decrement in the case
        of negative values) between consecutive elements in the ``RangeIndex``.

        See Also
        --------
        RangeIndex : Immutable index implementing a range-based index.
        RangeIndex.stop : Returns the stop value of the RangeIndex.
        RangeIndex.start : Returns the start value of the RangeIndex.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.step
        1

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.step
        -3

        Even if :class:`pandas.RangeIndex` is empty, ``step`` is still ``1`` if
        not supplied.

        >>> idx = pd.RangeIndex(1, 0)
        >>> idx.step
        1
        """
        # GH 25710
        return self._range.step

    @cache_readonly
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Cached property: number of bytes used, computed as the size of the range object
        # plus the sizes of its start, stop and step integers.
        """
        Return the number of bytes in the underlying data.
        """
        rng = self._range
        return getsizeof(rng) + sum(
            getsizeof(getattr(rng, attr_name))
            for attr_name in ["start", "stop", "step"]
        )

    def memory_usage(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # Return the memory usage in bytes. Parameter deep is accepted for API
        # compatibility and ignored. Always returns the same value as the byte-size
        # property.
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        # [FUNCTION DOC]
        # Property: always the int64 dtype object.
        return _dtype_int64

    @property
    def is_unique(self) -> bool:
        # [FUNCTION DOC]
        # Property: always true, since a nonzero-step range never repeats values.
        """return if the index has unique values"""
        return True

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: true when the step is positive or the length is at most one.
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: true when the step is negative or the length is at most one.
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        # [FUNCTION DOC]
        # Membership test. The key must be hashable (a type error is raised otherwise,
        # through the hash call). The key is converted to a Python integer; if this
        # fails with a type or overflow error the answer is false. Otherwise returns
        # whether the integer is in the range (constant time).
        hash(key)
        try:
            key = ensure_python_int(key)
        except (TypeError, OverflowError):
            return False
        return key in self._range

    @property
    def inferred_type(self) -> str:
        # [FUNCTION DOC]
        # Property: constant string "integer".
        return "integer"

    # --------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key) -> int:
        # [FUNCTION DOC]
        # Return the position of a label. Parameter key: an integer or a float with an
        # integer value (converted to int). Returns the integer position. Raises a key
        # error when the integer is not in the range or the key is any other hashable
        # value (non-integer floats, strings); for unhashable keys the indexing-error
        # check raises its invalid-index error, with a key error as the final fallback.
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : int or float
            Label to locate. Integer-like floats (e.g. 3.0) are accepted and
            treated as the corresponding integer. Non-integer floats and other
            non-integer labels are not valid and will raise KeyError or
            InvalidIndexError.

        Returns
        -------
        int
            Integer location of the label within the RangeIndex.

        Raises
        ------
        KeyError
            If the label is not present in the RangeIndex or the label is a
            non-integer value.
        InvalidIndexError
            If the label is of an invalid type for the RangeIndex.

        See Also
        --------
        RangeIndex.get_slice_bound : Calculate slice bound that corresponds to
            given label.
        RangeIndex.get_indexer : Computes indexer and mask for new index given
            the current index.
        RangeIndex.get_non_unique : Returns indexer and masks for new index given
            the current index.
        RangeIndex.get_indexer_for : Returns an indexer even when non-unique.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.get_loc(3)
        3

        >>> idx = pd.RangeIndex(2, 10, 2)  # values [2, 4, 6, 8]
        >>> idx.get_loc(6)
        2
        """
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key = int(key)
            try:
                return self._range.index(new_key)
            except ValueError as err:
                raise KeyError(key) from err
        if isinstance(key, Hashable):
            raise KeyError(key)
        self._check_indexing_error(key)
        raise KeyError(key)

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: computes, for each target label, its position in this index or minus
        # one if absent. Parameters: target is an Index; method, limit and tolerance are
        # fill options (defaults none). If any of them is given, the base class
        # implementation is used. Otherwise the arithmetic is done directly: for a
        # negative step the reversed range is used (GH 28678); the offset from the
        # start is computed, a label is valid if the offset is a multiple of the step,
        # nonnegative, and below the stop; positions are offset divided by step, and for
        # the reversed case mapped back to the original order. Returns a platform
        # integer array.
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(
                target, method=method, tolerance=tolerance, limit=limit
            )

        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            # GH 28678: work on reversed range for simplicity
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step

        target_array = np.asarray(target)
        locs = target_array - start
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step

        if step != self.step:
            # We reversed this range: transform to original locs
            locs[valid] = len(self) - 1 - locs[valid]
        return ensure_platform_int(locs)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached property saying integer keys should not be treated as
        # positional when indexing; always false.
        """
        Should an integer key be treated as positional?
        """
        return False

    # --------------------------------------------------------------------

    def tolist(self) -> list[int]:
        # [FUNCTION DOC]
        # Return the values as a Python list of integers, built from the range.
        return list(self._range)

    def __iter__(self) -> Iterator[int]:
        # [FUNCTION DOC]
        # Iterate over the values as Python integers by yielding from the range.
        """
        Return an iterator of the values.

        Returns
        -------
        iterator
            An iterator yielding ints from the RangeIndex.

        Examples
        --------
        >>> idx = pd.RangeIndex(3)
        >>> for x in idx:
        ...     print(x)
        0
        1
        2
        """
        yield from self._range

    def _shallow_copy(self, values, name: Hashable = no_default):
        # [FUNCTION DOC]
        # HELPER: creates a new index from values, sharing no data. Parameters: values
        # is an array with a dtype; name defaults to this index's name when left as the
        # no-default sentinel. Float-kind values give a float64 Index. One-dimensional
        # integer values give a RangeIndex when they form an arithmetic progression
        # (a single value uses this index's step; otherwise the base-index helper tries
        # to detect a range, GH 46675 and 43885); everything else becomes a result of
        # the constructor class (an Index). Returns the new index. The docstring says
        # the method is internal.
        """
        Create a new RangeIndex with the same class as the caller, don't copy the
        data, use the same object attributes with passed in attributes taking
        precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new RangeIndex, optional
        name : Label, defaults to self.name
        """
        name = self._name if name is no_default else name

        if values.dtype.kind == "f":
            return Index(values, name=name, dtype=np.float64, copy=False)
        if values.dtype.kind == "i" and values.ndim == 1:
            # GH 46675 & 43885: If values is equally spaced, return a
            # more memory-compact RangeIndex instead of Index with 64-bit dtype
            if len(values) == 1:
                start = values[0]
                new_range = range(start, start + self.step, self.step)
                return type(self)._simple_new(new_range, name=name)
            maybe_range = ibase.maybe_sequence_to_range(values)
            if isinstance(maybe_range, range):
                return type(self)._simple_new(maybe_range, name=name)
        return self._constructor._simple_new(values, name=name)

    def _view(self) -> Self:
        # [FUNCTION DOC]
        # HELPER: creates a new view of this index over the same range and name, sharing
        # the same cache dictionary, and registers it with the shared reference tracker.
        # Returns the view.
        result = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        self._references.add_index_reference(result)
        return result

    def _wrap_reindex_result(self, target, indexer, preserve_names: bool):
        # [FUNCTION DOC]
        # HELPER: post-processing of reindex results. If the target is not a RangeIndex
        # but has integer kind, it is first converted with the shallow-copy helper (which
        # may yield a RangeIndex). Then defers to the base class with the indexer and the
        # preserve_names flag. Returns the base result.
        if not isinstance(target, type(self)) and target.dtype.kind == "i":
            target = self._shallow_copy(target._values, name=target.name)
        return super()._wrap_reindex_result(target, indexer, preserve_names)

    def copy(self, name: Hashable | None = None, deep: bool = False) -> Self:
        # [FUNCTION DOC]
        # Return a copy with a possibly new name. Parameters: name (default none, meaning
        # keep the existing name); deep (default false) is only used for the name copy,
        # since the range is immutable. Returns a new RangeIndex through the rename
        # helper. The name argument is validated with the base class name validator.
        """
        Make a copy of this object.

        Name is set on the new object.

        Parameters
        ----------
        name : Label, optional
            Set name for new object.
        deep : bool, default False
            If True attempts to make a deep copy of the RangeIndex.
                Else makes a shallow copy.

        Returns
        -------
        RangeIndex
            RangeIndex refer to new object which is a copy of this object.

        See Also
        --------
        RangeIndex.delete: Make new RangeIndex with passed location(-s) deleted.
        RangeIndex.drop: Make new RangeIndex with passed list of labels deleted.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.

        Examples
        --------
        >>> idx = pd.RangeIndex(3)
        >>> new_idx = idx.copy()
        >>> idx is new_idx
        False
        """
        name = self._validate_names(name=name, deep=deep)[0]
        new_index = self._rename(name=name)
        return new_index

    def _minmax(self, meth: Literal["min", "max"]) -> int | float:
        # [FUNCTION DOC]
        # HELPER: shared minimum/maximum implementation. Parameter meth is the string min
        # or max. Returns not-a-number for an empty index; the start when asking for the
        # minimum with a positive step or the maximum with a negative step; otherwise the
        # last element computed arithmetically. Returns an integer or float.
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start

        return self.start + self.step * no_steps

    def min(self, axis=None, skipna: bool = True, *args, **kwargs) -> int | float:
        # [FUNCTION DOC]
        # Return the smallest value. Parameters: axis (validated to be none or zero),
        # skipna (accepted and ignored since there are no missing values), plus extra
        # positional and keyword arguments validated by the numpy compatibility layer
        # (an error is raised for unsupported ones). Returns an integer, or not-a-number
        # for an empty index.
        """The minimum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax("min")

    def max(self, axis=None, skipna: bool = True, *args, **kwargs) -> int | float:
        # [FUNCTION DOC]
        # Return the largest value. Same parameters and behavior as the minimum method.
        # Returns an integer, or not-a-number for an empty index.
        """The maximum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax("max")

    def _argminmax(
        self,
        meth: Literal["min", "max"],
        axis=None,
        skipna: bool = True,
    ) -> int:
        # [FUNCTION DOC]
        # HELPER: shared position-of-minimum/maximum implementation. Parameters: meth is
        # min or max; axis (validated none or zero); skipna. For an empty index it
        # delegates to the base class method (which raises). The minimum is at position
        # zero for a positive step, otherwise at the last position; the maximum is the
        # opposite.
        # KNOWN ISSUE: the final branch raising a value error for any other method name is unreachable for the two callers, which only pass min or max.
        nv.validate_minmax_axis(axis)
        if len(self) == 0:
            return getattr(super(), f"arg{meth}")(
                axis=axis,
                skipna=skipna,
            )
        elif meth == "min":
            if self.step > 0:
                return 0
            else:
                return len(self) - 1
        elif meth == "max":
            if self.step > 0:
                return len(self) - 1
            else:
                return 0
        else:
            raise ValueError(f"{meth=} must be max or min")

    def argmin(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        # [FUNCTION DOC]
        # Return the position of the smallest value. Parameters: axis, skipna as for the
        # helper; extra positional and keyword arguments are validated by the numpy
        # compatibility layer. Returns an integer position.
        nv.validate_argmin(args, kwargs)
        return self._argminmax("min", axis=axis, skipna=skipna)

    def argmax(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        # [FUNCTION DOC]
        # Return the position of the largest value. Same parameters as the position-of-
        # minimum method. Returns an integer position.
        nv.validate_argmax(args, kwargs)
        return self._argminmax("max", axis=axis, skipna=skipna)

    def argsort(self, *args, **kwargs) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Return the integer positions that would sort the values. Extra keyword
        # arguments: ascending (default true, accepted for extension-array compatibility)
        # and kind (ignored); the remaining positional and keyword arguments are
        # validated by the numpy compatibility layer. Returns a platform integer array
        # produced directly from arithmetic: the natural order for ascending with a
        # positive step or descending with a negative step, else reversed.
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
        ascending = kwargs.pop("ascending", True)  # EA compat
        kwargs.pop("kind", None)  # e.g. "mergesort" is irrelevant
        nv.validate_argsort(args, kwargs)

        start, stop, step = None, None, None
        if self._range.step > 0:
            if ascending:
                start = len(self)
            else:
                start, stop, step = len(self) - 1, -1, -1
        elif ascending:
            start, stop, step = len(self) - 1, -1, -1
        else:
            start = len(self)

        return np.arange(start, stop, step, dtype=np.intp)

    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> tuple[npt.NDArray[np.intp], RangeIndex]:
        # [FUNCTION DOC]
        # Factorize the values. Parameters: sort (default false) and use_na_sentinel
        # (default true, unused since there are no missing values). Returns a pair: the
        # integer codes, and the unique values. With sort true and a negative step the
        # codes count downward and the uniques are the reversed index; otherwise the
        # codes count upward and the uniques are this index itself.
        if sort and self.step < 0:
            codes = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
            uniques = self[::-1]
        else:
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self
        return codes, uniques

    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality of contents. Parameter other: any object. If other is a RangeIndex the
        # ranges are compared directly (two empty ranges compare equal); otherwise defers
        # to the base class comparison. Returns a boolean.
        """
        Determines if two Index objects contain the same elements.
        """
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return super().equals(other)

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[False] = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self: ...
        # [FUNCTION DOC]
        # Typing overload only: when the return indexer flag is false (default) the
        # result is only the sorted index. The implementation follows below.

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> tuple[Self, np.ndarray]: ...
        # [FUNCTION DOC]
        # Typing overload only: when the return indexer flag is true a pair of the sorted
        # index and the indexer array is returned. The implementation follows below.

    @overload
    def sort_values(
        self,
        *,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self | tuple[Self, np.ndarray]: ...
        # [FUNCTION DOC]
        # Typing overload only: for a boolean flag of unknown value either result form
        # is possible. The implementation follows below.

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable | None = None,
    ) -> Self | tuple[Self, np.ndarray]:
        # [FUNCTION DOC]
        # Sort the values. All parameters are keyword-only: return_indexer (default
        # false); ascending (default true); na_position (accepted, unused except when
        # delegating); key (callable, default none). If a key is given the base
        # implementation is used. Otherwise the result is this index itself or its
        # reverse, depending on whether the step direction agrees with the requested
        # order. Returns the sorted RangeIndex, or when return_indexer is true a pair of
        # the sorted index and the integer position array (reverse order if reversed,
        # natural order if not).
        if key is not None:
            return super().sort_values(
                return_indexer=return_indexer,
                ascending=ascending,
                na_position=na_position,
                key=key,
            )
        else:
            sorted_index = self
            inverse_indexer = False
            if ascending:
                if self.step < 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True
            elif self.step > 0:
                sorted_index = self[::-1]
                inverse_indexer = True

        if return_indexer:
            if inverse_indexer:
                indexer = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
            else:
                indexer = np.arange(len(self), dtype=np.intp)
            return sorted_index, indexer
        else:
            return sorted_index

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other: Index, sort: bool = False):
        # [FUNCTION DOC]
        # HELPER: intersection with another index. Parameters: other (caller guarantees
        # both are non-empty); sort (default false, ignored by the fast path). If other is
        # not a RangeIndex the base implementation is used. Otherwise both ranges are
        # normalized to increasing order, the overlapping interval is computed (empty
        # range returned if none), and a linear Diophantine problem solved with the
        # extended Euclid helper: if the starts are not congruent modulo the gcd the
        # result is empty; otherwise the new step is the least common multiple of the
        # steps and the first fitting element at or above the overlap start is found. The
        # result is reversed if both inputs were decreasing. Returns a RangeIndex without
        # name handling.
        # caller is responsible for checking self and other are both non-empty

        if not isinstance(other, RangeIndex):
            return super()._intersection(other, sort=sort)

        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range

        # check whether intervals intersect
        # deals with in- and decreasing ranges
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            return self._simple_new(_empty_range)

        # Method hint: linear Diophantine equation
        # solve intersection problem
        # performance hint: for identical step sizes, could use
        # cheaper alternative
        gcd, s, _ = self._extended_gcd(first.step, second.step)

        # check whether element sets intersect
        if (first.start - second.start) % gcd:
            return self._simple_new(_empty_range)

        # calculate parameters for the RangeIndex describing the
        # intersection disregarding the lower bounds
        tmp_start = first.start + (second.start - first.start) * first.step // gcd * s
        new_step = first.step * second.step // gcd

        # adjust index to limiting interval
        new_start = min_fitting_element(tmp_start, new_step, int_low)
        new_range = range(new_start, int_high, new_step)

        if (self.step < 0 and other.step < 0) is not (new_range.step < 0):
            new_range = new_range[::-1]

        return self._simple_new(new_range)

    def _extended_gcd(self, a: int, b: int) -> tuple[int, int, int]:
        # [FUNCTION DOC]
        # HELPER: extended Euclidean algorithm. Parameters a and b are integers. Returns
        # a triple of the gcd and the Bezout coefficients for a and b.
        # KNOWN ISSUE: the docstring writes the identity in terms of x and y (a*x + b*y = gcd(x, y)), but the gcd is of a and b.
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Finds one particular solution for x, y: s, t
        Returns: gcd, s, t
        """
        s, old_s = 0, 1
        t, old_t = 1, 0
        r, old_r = b, a
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        return old_r, old_s, old_t

    def _range_in_self(self, other: range) -> bool:
        # [FUNCTION DOC]
        # HELPER: tests whether a range is contained in this index's range. Parameter
        # other is a builtin range. Empty other is always contained; a non-empty other
        # is not contained in an empty self; if other has more than one element its step
        # must be a multiple of this step; then both its first and last elements must
        # be in this range. Returns a boolean.
        """Check if other range is contained in self"""
        # https://stackoverflow.com/a/32481015
        if not other:
            return True
        if not self._range:
            return False
        if len(other) > 1 and other.step % self._range.step:
            return False
        return other.start in self._range and other[-1] in self._range

    def _union(self, other: Index, sort: bool | None):
        # [FUNCTION DOC]
        # HELPER: union with another index. Parameters: other; sort (true, false or none).
        # Fast path only if other is a RangeIndex and either sort is none or true, or sort
        # is false while this index is increasing and contains the other range
        # (GH 47557). Both ranges are normalized to increasing order and the cases of
        # equal steps (adjacent/overlapping, or interleaved with half steps), or one step
        # a multiple of the other with compatible alignment, produce a single new
        # RangeIndex covering from the lowest start to the highest end. Single-element
        # ranges adopt the other's step (or their distance if both are single).
        # Everything else falls to the base class union. Returns the index.
        """
        Form the union of two Index objects and sorts if possible

        Parameters
        ----------
        other : Index or array-like

        sort : bool or None, default None
            Whether to sort (monotonically increasing) the resulting index.
            ``sort=None|True`` returns a ``RangeIndex`` if possible or a sorted
            ``Index`` with an int64 dtype if not.
            ``sort=False`` can return a ``RangeIndex`` if self is monotonically
            increasing and other is fully contained in self. Otherwise, returns
            an unsorted ``Index`` with an int64 dtype.

        Returns
        -------
        union : Index
        """
        if isinstance(other, RangeIndex):
            if sort in (None, True) or (
                sort is False and self.step > 0 and self._range_in_self(other._range)
            ):
                # GH 47557: Can still return a RangeIndex
                # if other range in self and sort=False
                start_s, step_s = self.start, self.step
                end_s = self.start + self.step * (len(self) - 1)
                start_o, step_o = other.start, other.step
                end_o = other.start + other.step * (len(other) - 1)
                if self.step < 0:
                    start_s, step_s, end_s = end_s, -step_s, start_s
                if other.step < 0:
                    start_o, step_o, end_o = end_o, -step_o, start_o
                if len(self) == 1 and len(other) == 1:
                    step_s = step_o = abs(self.start - other.start)
                elif len(self) == 1:
                    step_s = step_o
                elif len(other) == 1:
                    step_o = step_s
                start_r = min(start_s, start_o)
                end_r = max(end_s, end_o)
                if step_o == step_s:
                    if (
                        (start_s - start_o) % step_s == 0
                        and (start_s - end_o) <= step_s
                        and (start_o - end_s) <= step_s
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                    if (
                        (step_s % 2 == 0)
                        and (abs(start_s - start_o) == step_s / 2)
                        and (abs(end_s - end_o) == step_s / 2)
                    ):
                        # e.g. range(0, 10, 2) and range(1, 11, 2)
                        #  but not range(0, 20, 4) and range(1, 21, 4) GH#44019
                        return type(self)(start_r, end_r + step_s / 2, step_s / 2)

                elif step_o % step_s == 0:
                    if (
                        (start_o - start_s) % step_s == 0
                        and (start_o + step_s >= start_s)
                        and (end_o - step_s <= end_s)
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                elif step_s % step_o == 0:
                    if (
                        (start_s - start_o) % step_o == 0
                        and (start_s + step_o >= start_o)
                        and (end_s - step_o <= end_o)
                    ):
                        return type(self)(start_r, end_r + step_o, step_o)

        return super()._union(other, sort=sort)

    def _difference(self, other, sort=None):
        # [FUNCTION DOC]
        # HELPER: difference with another index. Parameters: other; sort (default none).
        # Validates the sort keyword, checks setop compatibility and converts other. For a
        # non-range other the base implementation is used. For a negative step with
        # sorting allowed, it works on the reverse. Computes the overlap with an
        # intersection: no overlap returns this index renamed with the result name; full
        # overlap returns an empty slice renamed; single-element overlaps at an end (or
        # the middle of a three element index) are sliced off; a two-element overlap at
        # both ends drops both ends; equal-step overlaps at the start, end or exactly
        # the interior give a new range; overlaps with twice the step at alternating
        # positions give a stepped slice; every other pattern is not range-like and
        # falls to the base class. Returns an index with the combined result name.
        # optimized set operation if we have another RangeIndex
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        if not isinstance(other, RangeIndex):
            return super()._difference(other, sort=sort)

        if sort is not False and self.step < 0:
            return self[::-1]._difference(other)

        res_name = ops.get_op_result_name(self, other)

        first = self._range[::-1] if self.step < 0 else self._range
        overlap = self.intersection(other)
        if overlap.step < 0:
            overlap = overlap[::-1]

        if len(overlap) == 0:
            return self.rename(name=res_name)
        if len(overlap) == len(self):
            return self[:0].rename(res_name)

        # overlap.step will always be a multiple of self.step (see _intersection)

        if len(overlap) == 1:
            if overlap[0] == self[0]:
                return self[1:]

            elif overlap[0] == self[-1]:
                return self[:-1]

            elif len(self) == 3 and overlap[0] == self[1]:
                return self[::2]

            else:
                return super()._difference(other, sort=sort)

        elif len(overlap) == 2 and overlap[0] == first[0] and overlap[-1] == first[-1]:
            # e.g. range(-8, 20, 7) and range(13, -9, -3)
            return self[1:-1]

        if overlap.step == first.step:
            if overlap[0] == first.start:
                # The difference is everything after the intersection
                new_rng = range(overlap[-1] + first.step, first.stop, first.step)
            elif overlap[-1] == first[-1]:
                # The difference is everything before the intersection
                new_rng = range(first.start, overlap[0], first.step)
            elif overlap._range == first[1:-1]:
                # e.g. range(4) and range(1, 3)
                step = len(first) - 1
                new_rng = first[::step]
            else:
                # The difference is not range-like
                # e.g. range(1, 10, 1) and range(3, 7, 1)
                return super()._difference(other, sort=sort)

        else:
            # We must have len(self) > 1, bc we ruled out above
            #  len(overlap) == 0 and len(overlap) == len(self)
            assert len(self) > 1

            if overlap.step == first.step * 2:
                if overlap[0] == first[0] and overlap[-1] in (first[-1], first[-2]):
                    # e.g. range(1, 10, 1) and range(1, 10, 2)
                    new_rng = first[1::2]

                elif overlap[0] == first[1] and overlap[-1] in (first[-1], first[-2]):
                    # e.g. range(1, 10, 1) and range(2, 10, 2)
                    new_rng = first[::2]

                else:
                    # We can get here with  e.g. range(20) and range(0, 10, 2)
                    return super()._difference(other, sort=sort)

            else:
                # e.g. range(10) and range(0, 10, 3)
                return super()._difference(other, sort=sort)

        if first is not self._range:
            new_rng = new_rng[::-1]
        new_index = type(self)._simple_new(new_rng, name=res_name)

        return new_index

    def symmetric_difference(
        self, other, result_name: Hashable | None = None, sort=None
    ) -> Index:
        # [FUNCTION DOC]
        # Symmetric difference. Parameters: other; result_name (default none); sort
        # (default none). If other is not a RangeIndex or a sort is requested, the base
        # class is used. Otherwise returns the union of the two one-sided differences,
        # renamed when a result name is given.
        if not isinstance(other, RangeIndex) or sort is not None:
            return super().symmetric_difference(other, result_name, sort)

        left = self.difference(other)
        right = other.difference(self)
        result = left.union(right)

        if result_name is not None:
            result = result.rename(result_name)
        return result

    def _join_empty(
        self, other: Index, how: JoinHow, sort: bool
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # [FUNCTION DOC]
        # HELPER: join when one side is empty. Parameters: other, how, sort. If other is
        # a non-range integer index it is first converted to a RangeIndex when possible
        # via the shallow-copy helper. Then defers to the base class and returns its
        # triple of the joined index and the two optional indexers.
        if not isinstance(other, RangeIndex) and other.dtype.kind == "i":
            other = self._shallow_copy(other._values, name=other.name)
        return super()._join_empty(other, how=how, sort=sort)

    def _join_monotonic(
        self, other: Index, how: JoinHow = "left"
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # [FUNCTION DOC]
        # HELPER: join for monotonic increasing indexes. Parameters: other and how
        # (left, right, inner or outer; default left). If other is not a RangeIndex it is
        # converted when possible, otherwise the base class is used. Identical indexes
        # return one of them with no indexers. Otherwise the join index is this index
        # (left), other (right), the intersection (inner) or the union (outer), with
        # indexers obtained by looking up the join index in each side (none when that
        # side is the join index itself). Returns a triple of the join index and two
        # platform integer indexers or none.
        # This currently only gets called for the monotonic increasing case
        if not isinstance(other, type(self)):
            maybe_ri = self._shallow_copy(other._values, name=other.name)
            if not isinstance(maybe_ri, type(self)):
                return super()._join_monotonic(other, how=how)
            other = maybe_ri

        if self.equals(other):
            ret_index = other if how == "right" else self
            return ret_index, None, None

        if how == "left":
            join_index = self
            lidx = None
            ridx = other.get_indexer(join_index)
        elif how == "right":
            join_index = other
            lidx = self.get_indexer(join_index)
            ridx = None
        elif how == "inner":
            join_index = self.intersection(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)
        elif how == "outer":
            join_index = self.union(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)

        lidx = None if lidx is None else ensure_platform_int(lidx)
        ridx = None if ridx is None else ensure_platform_int(ridx)
        return join_index, lidx, ridx

    # --------------------------------------------------------------------

    # error: Return type "Index" of "delete" incompatible with return type
    #  "RangeIndex" in supertype "Index"
    def delete(self, loc) -> Index:  # type: ignore[override]
        # [FUNCTION DOC]
        # Delete the element or elements at the given position(s). Parameter loc is an
        # integer or list-like of integers. Single integer positions at either end (or
        # the middle of a three element index) return a RangeIndex by slicing; a
        # list-like that can be expressed as a slice is deleted by taking the difference
        # with the corresponding slice (which also keeps a RangeIndex when possible).
        # Everything else falls to the base class (which returns a plain Index and raises
        # index errors for out-of-bounds positions).
        # In some cases we can retain RangeIndex, see also
        #  DatetimeTimedeltaMixin._get_delete_Freq
        if is_integer(loc):
            if loc in (0, -len(self)):
                return self[1:]
            if loc in (-1, len(self) - 1):
                return self[:-1]
            if len(self) == 3 and loc in (1, -2):
                return self[::2]

        elif lib.is_list_like(loc):
            slc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))

            if isinstance(slc, slice):
                # defer to RangeIndex._difference, which is optimized to return
                #  a RangeIndex whenever possible
                other = self[slc]
                return self.difference(other, sort=False)

        return super().delete(loc)

    def insert(self, loc: int, item) -> Index:
        # [FUNCTION DOC]
        # Insert an item before a position. Parameters: loc (integer position) and item.
        # A RangeIndex is retained when inserting an integer or float that continues the
        # progression at the start or end (by exactly one step), into an empty index at
        # position zero (integer only), or midway in a two-element index (half step).
        # Otherwise defers to the base class, which returns a plain Index. Returns the
        # new index.
        if is_integer(item) or is_float(item):
            # We can retain RangeIndex is inserting at the beginning or end,
            #  or right in the middle.
            if len(self) == 0 and loc == 0 and is_integer(item):
                new_rng = range(item, item + self.step, self.step)
                return type(self)._simple_new(new_rng, name=self._name)
            elif len(self):
                rng = self._range
                if loc == 0 and item == self[0] - self.step:
                    new_rng = range(rng.start - rng.step, rng.stop, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)

                elif loc == len(self) and item == self[-1] + self.step:
                    new_rng = range(rng.start, rng.stop + rng.step, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)

                elif len(self) == 2 and item == self[0] + self.step / 2:
                    # e.g. inserting 1 into [0, 2]
                    step = int(self.step / 2)
                    new_rng = range(self.start, self.stop, step)
                    return type(self)._simple_new(new_rng, name=self._name)

        return super().insert(loc, item)

    def _concat(self, indexes: list[Index], name: Hashable) -> Index:
        # [FUNCTION DOC]
        # HELPER: concatenation of indexes. Parameters: indexes is a list of indexes;
        # name is the result name. If not all are RangeIndex, the base class result is
        # converted back to a RangeIndex when it has integer dtype. A single index is
        # returned as is. Otherwise empty ranges are ignored and the remaining ranges are
        # checked to be consecutive with a common step; the result is a RangeIndex
        # spanning them (or empty if all are empty). Non-consecutive cases build an
        # integer Index from the concatenated values (tiling when all non-empty inputs
        # are identical), renamed with the result name. The special case where the first
        # non-empty range has a single element is also handled.
        """
        Overriding parent method for the case of all RangeIndex instances.

        When all members of "indexes" are of type RangeIndex: result will be
        RangeIndex if possible, Index with an int64 dtype otherwise. E.g.:
        indexes = [RangeIndex(3), RangeIndex(3, 6)] -> RangeIndex(6)
        indexes = [RangeIndex(3), RangeIndex(4, 6)] -> Index([0,1,2,4,5], dtype='int64')
        """
        if not all(isinstance(x, RangeIndex) for x in indexes):
            result = super()._concat(indexes, name)
            if result.dtype.kind == "i":
                return self._shallow_copy(result._values)
            return result

        elif len(indexes) == 1:
            return indexes[0]

        rng_indexes = cast(list[RangeIndex], indexes)

        start = step = next_ = None

        # Filter the empty indexes
        non_empty_indexes = []
        all_same_index = True
        prev: RangeIndex | None = None
        for obj in rng_indexes:
            if len(obj):
                non_empty_indexes.append(obj)
                if all_same_index:
                    if prev is not None:
                        all_same_index = prev.equals(obj)
                    else:
                        prev = obj

        for obj in non_empty_indexes:
            rng = obj._range

            if start is None:
                # This is set by the first non-empty index
                start = rng.start
                if step is None and len(rng) > 1:
                    step = rng.step
            elif step is None:
                # First non-empty index had only one element
                if rng.start == start:
                    if all_same_index:
                        values = np.tile(
                            non_empty_indexes[0]._values, len(non_empty_indexes)
                        )
                    else:
                        values = np.concatenate([x._values for x in rng_indexes])
                    result = self._constructor(values, copy=False)
                    return result.rename(name)

                step = rng.start - start

            non_consecutive = (step != rng.step and len(rng) > 1) or (
                next_ is not None and rng.start != next_
            )
            if non_consecutive:
                if all_same_index:
                    values = np.tile(
                        non_empty_indexes[0]._values, len(non_empty_indexes)
                    )
                else:
                    values = np.concatenate([x._values for x in rng_indexes])
                result = self._constructor(values, copy=False)
                return result.rename(name)

            if step is not None:
                next_ = rng[-1] + step

        if non_empty_indexes:
            # Get the stop value from "next" or alternatively
            # from the last non-empty index
            stop = non_empty_indexes[-1].stop if next_ is None else next_
            if len(non_empty_indexes) == 1:
                step = non_empty_indexes[0].step
            return RangeIndex(start, stop, step, name=name)

        # Here all "indexes" had 0 length, i.e. were empty.
        # In this case return an empty range index.
        return RangeIndex(_empty_range, name=name)

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Return the number of values (the length of the range).
        """
        return the length of the RangeIndex
        """
        return len(self._range)

    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property: the number of values, equal to the length.
        return len(self)

    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Item access. Parameter key: integer, slice, ellipsis, zero-dimensional array,
        # boolean mask or array of integers. Slices go through the slice fast path and
        # return a RangeIndex. Integers return a Python integer, raising an index error
        # with an out-of-bounds message. Other scalars raise an index error listing the
        # valid index kinds. Boolean masks (including extension-boolean arrays, with
        # missing treated as false) are validated and converted to positions. List-like
        # positions are passed to the take method; if that fails with a type or value
        # error the base class handles the key.
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
        key = lib.item_from_zerodim(key)
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            return self._getitem_slice(key)
        elif is_integer(key):
            new_key = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {len(self)}"
                ) from err
        elif is_scalar(key):
            raise IndexError(
                "only integers, slices (`:`), "
                "ellipsis (`...`), numpy.newaxis (`None`) "
                "and integer or boolean "
                "arrays are valid indices"
            )
        elif com.is_bool_indexer(key):
            if isinstance(getattr(key, "dtype", None), ExtensionDtype):
                key = key.to_numpy(dtype=bool, na_value=False)
            else:
                key = np.asarray(key, dtype=bool)
            check_array_indexer(self._range, key)  # type: ignore[arg-type]
            key = np.flatnonzero(key)
        try:
            return self.take(key)
        except (TypeError, ValueError):
            return super().__getitem__(key)

    def _getitem_slice(self, slobj: slice) -> Self:
        # [FUNCTION DOC]
        # HELPER: slice fast path. Parameter slobj is a slice. Slices the range and wraps
        # the result in a RangeIndex that keeps the name.
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._range[slobj]
        return type(self)._simple_new(res, name=self._name)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        # [FUNCTION DOC]
        # Floor division by other. Zero-dimensional arguments are unpacked and
        # unsupported types deferred by the decorator. If other is a nonzero integer and
        # either the index is empty or both start and step are divisible by it, or the
        # index has a single element, the result is a RangeIndex with divided parameters
        # (a zero step becomes one). Otherwise defers to the base class which returns a
        # plain Index.
        if is_integer(other) and other != 0:
            if len(self) == 0 or (self.start % other == 0 and self.step % other == 0):
                start = self.start // other
                step = self.step // other
                stop = start + len(self) * step
                new_range = range(start, stop, step or 1)
                return self._simple_new(new_range, name=self._name)
            if len(self) == 1:
                start = self.start // other
                new_range = range(start, start + 1, 1)
                return self._simple_new(new_range, name=self._name)

        return super().__floordiv__(other)

    # --------------------------------------------------------------------
    # Reductions

    def all(self, *args, **kwargs) -> bool:
        # [FUNCTION DOC]
        # Reduction: true when no element is zero (i.e. zero is not in the range).
        # Arguments are ignored.
        return 0 not in self._range

    def any(self, *args, **kwargs) -> bool:
        # [FUNCTION DOC]
        # Reduction: true when any element is nonzero (evaluated over the range).
        # Arguments are ignored.
        return any(self._range)

    # --------------------------------------------------------------------

    # error: Return type "RangeIndex | Index" of "round" incompatible with
    # return type "RangeIndex" in supertype "Index"
    def round(self, decimals: int = 0) -> Self | Index:  # type: ignore[override]
        # [FUNCTION DOC]
        # Round the values to a number of decimals. Parameter decimals is an integer,
        # default zero. Non-negative decimals return a copy; negative decimals return a
        # copy if start and step are already multiples of ten to the minus decimals,
        # otherwise the base class rounds and returns a plain Index.
        """
        Round each value in the Index to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point
            e.g. ``round(11.0, -1) == 10.0``.

        Returns
        -------
        Index or RangeIndex
            A new Index with the rounded values.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.RangeIndex(10, 30, 10)
        >>> idx.round(decimals=-1)
        RangeIndex(start=10, stop=30, step=10)
        >>> idx = pd.RangeIndex(10, 15, 1)
        >>> idx.round(decimals=-1)
        Index([10, 10, 10, 10, 10], dtype='int64')
        """
        if decimals >= 0:
            return self.copy()
        elif self.start % 10**-decimals == 0 and self.step % 10**-decimals == 0:
            # e.g. RangeIndex(10, 30, 10).round(-1) doesn't need rounding
            return self.copy()
        else:
            return super().round(decimals=decimals)

    def _cmp_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: comparison operator hook. Parameters: other and op. If other is a
        # RangeIndex with an equal range, the shortcut passes this index itself as the
        # other operand to the base implementation; otherwise the base class is used with
        # the given other. Returns the base class result (a boolean array).
        if isinstance(other, RangeIndex) and self._range == other._range:
            # Both are immutable so if ._range attr. are equal, shortcut is possible
            return super()._cmp_method(self, op)
        return super()._cmp_method(other, op)

    def _arith_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: arithmetic operator hook. Parameters: other and op (a binary operator
        # function). Timedelta indexes are deferred (not-implemented is returned);
        # timedelta scalars and arrays go to the base class; power, modulo, floor
        # division and divmod (and reflected forms) go to the base class. Otherwise the
        # new start, stop and step are computed from the operand: multiplication and
        # division scale the step (the result must be a nonzero integer or the base class
        # is used), reflected subtraction negates it, and others keep it. The result is a
        # RangeIndex named from both operands; if any computed parameter is not an
        # integer it is converted to a float64 Index (GH#53255 and numpy compatibility).
        # A value error, type error or zero-division error in this process falls back to
        # the base class.
        """
        Parameters
        ----------
        other : Any
        op : callable that accepts 2 params
            perform the binary op
        """

        if isinstance(other, ABCTimedeltaIndex):
            # Defer to TimedeltaIndex implementation
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            # GH#19333 is_integer evaluated True on timedelta64,
            # so we need to catch these explicitly
            return super()._arith_method(other, op)
        elif lib.is_np_dtype(getattr(other, "dtype", None), "m"):
            # Must be an np.ndarray; GH#22390
            return super()._arith_method(other, op)

        if op in [
            operator.pow,
            ops.rpow,
            operator.mod,
            ops.rmod,
            operator.floordiv,
            ops.rfloordiv,
            divmod,
            ops.rdivmod,
        ]:
            return super()._arith_method(other, op)

        step: Callable | None = None
        if op in [operator.mul, ops.rmul, operator.truediv, ops.rtruediv]:
            step = op

        # TODO: if other is a RangeIndex we may have more efficient options
        right = extract_array(other, extract_numpy=True, extract_range=True)
        left = self

        try:
            # apply if we have an override
            if step:
                with np.errstate(all="ignore"):
                    rstep = step(left.step, right)

                # we don't have a representable op
                # so return a base index
                if not is_integer(rstep) or not rstep:
                    raise ValueError

            # GH#53255
            else:
                rstep = -left.step if op == ops.rsub else left.step

            with np.errstate(all="ignore"):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)

            res_name = ops.get_op_result_name(self, other)
            result = type(self)(rstart, rstop, rstep, name=res_name)

            # for compat with numpy / Index with int64 dtype
            # even if we can represent as a RangeIndex, return
            # as a float64 Index if we have float-like descriptors
            if not all(is_integer(x) for x in [rstart, rstop, rstep]):
                result = result.astype("float64")

            return result

        except (ValueError, TypeError, ZeroDivisionError):
            # test_arithmetic_explicit_conversions
            return super()._arith_method(other, op)

    def __abs__(self) -> Self | Index:
        # [FUNCTION DOC]
        # Absolute value. Returns a copy if empty or all non-negative, the negation if all
        # non-positive, and otherwise defers to the base class (plain Index).
        if len(self) == 0 or self.min() >= 0:
            return self.copy()
        elif self.max() <= 0:
            return -self
        else:
            return super().__abs__()

    def __neg__(self) -> Self:
        # [FUNCTION DOC]
        # Negation: returns a RangeIndex with negated start, stop and step, keeping the
        # name.
        rng = range(-self.start, -self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    def __pos__(self) -> Self:
        # [FUNCTION DOC]
        # Unary plus: returns a copy.
        return self.copy()

    def __invert__(self) -> Self:
        # [FUNCTION DOC]
        # Bitwise inversion of every value: an empty index is copied; otherwise a
        # RangeIndex is built from the bitwise complement of start and stop and a negated
        # step, keeping the name.
        if len(self) == 0:
            return self.copy()
        rng = range(~self.start, ~self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    # error: Return type "Index" of "take" incompatible with return type
    # "RangeIndex" in supertype "Index"
    def take(  # type: ignore[override]
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self | Index:
        # [FUNCTION DOC]
        # Positional take. Parameters: indices (array-like of integers; a scalar raises a
        # type error); axis (default 0, validated by base class conventions); allow_fill
        # (default true) and fill_value (default none); extra keyword arguments are
        # validated by the numpy compatibility layer. Raises a value error if fill is
        # requested with a non-none fill value (via the disallow-fill helper) and an
        # index error for any position out of bounds. Negative positions count from the
        # end. An empty indices array returns an empty RangeIndex. Positions are mapped
        # arithmetically (times step plus start) and wrapped with the shallow-copy helper
        # so the result is a RangeIndex when the taken values form a progression, else a
        # plain Index. The name is kept.
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)

        # raise an exception if allow_fill is True and fill_value is not None
        self._maybe_disallow_fill(allow_fill, fill_value, indices)

        if len(indices) == 0:
            return type(self)(_empty_range, name=self.name)
        else:
            ind_max = indices.max()
            if ind_max >= len(self):
                raise IndexError(
                    f"index {ind_max} is out of bounds for axis 0 with size {len(self)}"
                )
            ind_min = indices.min()
            if ind_min < -len(self):
                raise IndexError(
                    f"index {ind_min} is out of bounds for axis 0 with size {len(self)}"
                )
            taken = indices.astype(self.dtype, casting="safe")
            if ind_min < 0:
                taken %= len(self)
            if self.step != 1:
                taken *= self.step
            if self.start != 0:
                taken += self.start

        return self._shallow_copy(taken, name=self.name)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series:
        # [FUNCTION DOC]
        # Count the values. Parameters: normalize (default false); sort, ascending, dropna
        # (ignored on the fast path); bins (default none; if given the base class is
        # used). Otherwise returns a series where every value of this index maps to one
        # count (or the proportion one over the length when normalize is set); the
        # series is indexed by a copy of this index and named count or proportion.
        from pandas import Series

        if bins is not None:
            return super().value_counts(
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            )
        name = "proportion" if normalize else "count"
        data: npt.NDArray[np.floating] | npt.NDArray[np.signedinteger] = np.ones(
            len(self), dtype=np.int64
        )
        if normalize:
            data = data / len(self)
        return Series(data, index=self.copy(), name=name)

    @overload
    def searchsorted(  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
        self,
        value: ScalarLike_co,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> np.intp: ...
        # [FUNCTION DOC]
        # Typing overload only: scalar value yields a single integer position. The
        # implementation follows below.

    @overload
    def searchsorted(
        self,
        value: npt.ArrayLike | ExtensionArray,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> npt.NDArray[np.intp]: ...
        # [FUNCTION DOC]
        # Typing overload only: array-like value yields an array of integer positions.
        # The implementation follows below.

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Binary search for insertion positions. Parameters: value (scalar or array-like
        # or extension array); side (left or right; default left); sorter (default none).
        # If side is invalid or a sorter is provided, or the values are not integers, the
        # base class is used. Otherwise positions are computed arithmetically from the
        # start and step, flipping the computation for a negative step, and clipped to
        # the range zero to the length. Returns an integer position for scalar input,
        # otherwise an array of platform integers.
        if side not in {"left", "right"} or sorter is not None:
            return super().searchsorted(value=value, side=side, sorter=sorter)

        was_scalar = False
        if is_scalar(value):
            was_scalar = True
            array_value = np.array([value])
        else:
            array_value = np.asarray(value)
        if array_value.dtype.kind not in "iu":
            return super().searchsorted(value=value, side=side, sorter=sorter)

        if flip := (self.step < 0):
            rng = self._range[::-1]
            start = rng.start
            step = rng.step
            shift = side == "right"
        else:
            start = self.start
            step = self.step
            shift = side == "left"
        result = (array_value - start - int(shift)) // step + 1
        if flip:
            result = len(self) - result
        result = np.maximum(np.minimum(result, len(self)), 0)
        if was_scalar:
            return np.intp(result.item())
        return result.astype(np.intp, copy=False)
