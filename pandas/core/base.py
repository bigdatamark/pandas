"""
Base and utility classes for pandas objects.
"""

# [MODULE DOC]
# Base classes shared by the main containers. Public contents (all internal,
# not part of the user API): PandasObject (root class with cache handling and
# memory-size reporting), NoNewAttributesMixin (freezes an object against new
# attributes, used by accessors), SelectionMixin (column selection and
# aggregation interface for group-like objects such as groupby, rolling and
# resample), and IndexOpsMixin (the common interface of Series and Index:
# shape-related properties, conversion, unique/duplicate handling, value counts,
# factorization, searching, arithmetic dispatch, and abstract hooks).
# Depends on algorithms, nanops, ops, accessor and arraylike mixins and on the
# array classes. No import-time side effects beyond imports.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    cast,
    final,
    overload,
)

import numpy as np

from pandas._libs import lib
from pandas._typing import (
    AxisInt,
    DtypeObj,
    IndexLabel,
    NDFrameT,
    Shape,
    npt,
)
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import (
    is_object_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    remove_na_arraylike,
)

from pandas.core import (
    algorithms,
    nanops,
    ops,
)
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

    from pandas._typing import (
        DropKeep,
        NumpySorter,
        NumpyValueArrayLike,
        ScalarLike_co,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


class PandasObject(DirNamesMixin):
    # [CLASS DOC]
    # Root base class of pandas containers (subclass of the directory-listing
    # mixin). Declares a per-instance cache dict, filled by methods decorated as
    # cached read-only properties, and provides cache clearing, a default
    # representation, and a memory-size report.
    """
    Base class for various pandas objects.
    """

    # results from calls to methods decorated with cache_readonly get added to _cache
    _cache: dict[str, Any]

    @property
    def _constructor(self) -> type[Self]:
        # [FUNCTION DOC]
        # Property: the class used to build new instances of the same kind; here
        # simply the object's own class (subclasses override).
        """
        Class constructor (for this class it's just `__class__`).
        """
        return type(self)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Default representation: the plain object representation (class and
        # address). Subclasses are expected to override.
        """
        Return a string representation for a particular object.
        """
        # Should be overwritten by base classes
        return object.__repr__(self)

    def _reset_cache(self, key: str | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: invalidates cached property values. Input key (default none):
        # a single cache entry name to drop (missing names are ignored); none
        # clears the whole cache. Does nothing if the object has no cache dict.
        """
        Reset cached properties. If ``key`` is passed, only clears that key.
        """
        if not hasattr(self, "_cache"):
            return
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __sizeof__(self) -> int:
        # [FUNCTION DOC]
        # Reports the object's size in bytes. If the object has a memory-usage
        # method, calls it with deep inspection and returns the integer total (a
        # scalar result is used directly, a per-column result is summed);
        # otherwise falls back to the default object size.
        """
        Generates the total memory usage for an object that returns
        either a value or Series of values
        """
        memory_usage = getattr(self, "memory_usage", None)
        if memory_usage:
            mem = memory_usage(deep=True)
            return int(mem if is_scalar(mem) else mem.sum())

        # no memory_usage attribute, so fall back to object's 'sizeof'
        return super().__sizeof__()


class NoNewAttributesMixin:
    # [CLASS DOC]
    # Mixin that can lock an object against gaining new attributes (used by
    # accessors). State: a hidden frozen marker set by the freeze method.
    # While frozen, assignment is allowed only for the cache attribute, names
    # defined on the class itself, or names that already resolve to a non-none
    # value on the instance; anything else raises an attribute error. The lock
    # can be bypassed by calling the base-object setter directly.
    """
    Mixin which prevents adding new attributes.

    Prevents additional attributes via xxx.attribute = "something" after a
    call to `self.__freeze()`. Mainly used to prevent the user from using
    wrong attributes on an accessor (`Series.cat/.str/.dt`).

    If you really want to add a new attribute at a later time, you need to use
    `object.__setattr__(self, key, value)`.
    """

    def _freeze(self) -> None:
        # [FUNCTION DOC]
        # Marks the instance as frozen (sets the hidden marker, bypassing the
        # restricted setter). Irreversible through normal assignment.
        """
        Prevents setting additional attributes.
        """
        object.__setattr__(self, "__frozen", True)

    # prevent adding any attribute via s.xxx.new_attribute = ...
    def __setattr__(self, key: str, value) -> None:
        # [FUNCTION DOC]
        # Attribute assignment with the frozen check. Inputs: key, value. If the
        # instance is frozen and the key is not the cache attribute, not in the
        # class's own namespace, and does not already resolve to a non-none value
        # on the instance, raises an attribute error saying new attributes cannot
        # be added. Otherwise stores the value normally.
        # _cache is used by a decorator
        # We need to check both 1.) cls.__dict__ and 2.) getattr(self, key)
        # because
        # 1.) getattr is false for attributes that raise errors
        # 2.) cls.__dict__ doesn't traverse into base classes
        if getattr(self, "__frozen", False) and not (
            key == "_cache"
            or key in type(self).__dict__
            or getattr(self, key, None) is not None
        ):
            raise AttributeError(f"You cannot add any new attribute '{key}'")
        object.__setattr__(self, key, value)


class SelectionMixin(Generic[NDFrameT]):
    # [CLASS DOC]
    # Mixin giving group-like objects (grouping, rolling, resampling) column
    # selection and an aggregation interface. Generic in the frame type.
    # Subclasses must provide obj (the wrapped Series or DataFrame) and
    # exclusions (set of labels to leave out); this class tracks the optional
    # current selection. Class attributes: the selection (default none) and
    # internal attribute names (the cache and unpickling hook) hidden from
    # directory listings. Subclasses must implement the sub-selection hook and
    # aggregate (both abstract here).
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """

    obj: NDFrameT
    _selection: IndexLabel | None = None
    exclusions: frozenset[Hashable]
    _internal_names = ["_cache", "__setstate__"]
    _internal_names_set = set(_internal_names)

    @final
    @property
    def _selection_list(self):
        # [FUNCTION DOC]
        # Property: the current selection as a list-like. A scalar selection (or
        # none) is wrapped in a one-element list; list, tuple, Series, Index and
        # array selections are returned unchanged.
        if not isinstance(
            self._selection, (list, tuple, ABCSeries, ABCIndex, np.ndarray)
        ):
            return [self._selection]
        return self._selection

    @cache_readonly
    def _selected_obj(self):
        # [FUNCTION DOC]
        # Cached property: the object restricted to the selection. Returns the
        # wrapped object itself when there is no selection or it is a Series;
        # otherwise the wrapped frame indexed by the selection.
        if self._selection is None or isinstance(self.obj, ABCSeries):
            return self.obj
        else:
            return self.obj[self._selection]

    @final
    @cache_readonly
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Cached property: number of dimensions of the selected object.
        return self._selected_obj.ndim

    @final
    @cache_readonly
    def _obj_with_exclusions(self):
        # [FUNCTION DOC]
        # Cached property: the wrapped object without excluded columns. A Series
        # is returned as-is. With a selection, the frame indexed by the selection
        # list. Otherwise, if there are exclusions, a column-sliced view dropping
        # them (avoiding consolidation and copying); else the wrapped frame.
        if isinstance(self.obj, ABCSeries):
            return self.obj

        if self._selection is not None:
            return self.obj[self._selection_list]

        if len(self.exclusions) > 0:
            # equivalent to `self.obj.drop(self.exclusions, axis=1)
            #  but this avoids consolidating and making a copy
            # TODO: following GH#45287 can we now use .drop directly without
            #  making a copy?
            return self.obj._drop_axis(self.exclusions, axis=1, only_slice=True)
        else:
            return self.obj

    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Selects column(s). Input key: a label, or a list, tuple, Series, Index
        # or array of labels. Raises an index error if a selection was already
        # made. For several labels: raises a key error listing missing columns,
        # otherwise returns the sub-selection hook's result with the labels as a
        # list and two dimensions. For a single label: raises a key error if
        # absent; otherwise returns the hook's result with that label and the
        # dimensionality of the corresponding column.
        if self._selection is not None:
            raise IndexError(f"Column(s) {self._selection} already selected")

        if isinstance(key, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            if len(self.obj.columns.intersection(key)) != len(set(key)):
                bad_keys = list(set(key).difference(self.obj.columns))
                raise KeyError(f"Columns not found: {str(bad_keys)[1:-1]}")
            return self._gotitem(list(key), ndim=2)

        else:
            if key not in self.obj:
                raise KeyError(f"Column not found: {key}")
            ndim = self.obj[key].ndim
            return self._gotitem(key, ndim=ndim)

    def _gotitem(self, key, ndim: int, subset=None):
        # [FUNCTION DOC]
        # HELPER: abstract hook that subclasses implement to return a sliced
        # object. Inputs: key, label or list of labels; ndim, requested
        # dimensionality (1 or 2); subset, optional object to act on. This base
        # version always raises the abstract-method error.
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        raise AbstractMethodError(self)

    @final
    def _infer_selection(self, key, subset: Series | DataFrame):
        # [FUNCTION DOC]
        # HELPER: computes the selection to pass to a subclass constructor from
        # the sub-selection hook (shared by rolling and resampling). Inputs: key;
        # subset, the object being sliced. For two-dimensional subsets, the key
        # is the selection if it is a scalar present in the subset or any
        # list-like. For one-dimensional subsets, the key is the selection if it
        # is a scalar equal to the subset's name. Otherwise none.
        """
        Infer the `selection` to pass to our constructor in _gotitem.
        """
        # Shared by Rolling and Resample
        selection = None
        if subset.ndim == 2 and (
            (lib.is_scalar(key) and key in subset) or lib.is_list_like(key)
        ):
            selection = key
        elif subset.ndim == 1 and lib.is_scalar(key) and key == subset.name:
            selection = key
        return selection

    def aggregate(self, func, *args, **kwargs):
        # [FUNCTION DOC]
        # Abstract aggregation entry point (alias: agg); always raises the
        # abstract-method error. Subclasses implement it.
        raise AbstractMethodError(self)

    agg = aggregate


class IndexOpsMixin(OpsMixin):
    # [CLASS DOC]
    # Common interface of one-dimensional labelled containers (Series and Index);
    # extends the operator mixin. Subclasses supply the dtype, the underlying
    # values (numpy or extension array), the length and the result constructor;
    # this mixin builds on those for shape properties, conversion to numpy and
    # lists, iteration, position of extremes, uniqueness and monotonicity,
    # counting, factorizing, searching, duplicate dropping and arithmetic
    # dispatch. Class attributes: a high array priority so numpy defers to it
    # in mixed operations, and the set of hidden attributes (the list-conversion
    # method is suppressed from directory listings).
    """
    Common ops mixin to support a unified interface / docs for Series / Index
    """

    # ndarray compatibility
    __array_priority__ = 1000
    _hidden_attrs: frozenset[str] = frozenset(
        ["tolist"]  # tolist is not deprecated, just suppressed in the __dir__
    )

    @property
    def dtype(self) -> DtypeObj:
        # [FUNCTION DOC]
        # Abstract property declared for type checkers; raises the
        # abstract-method error. Subclasses return their dtype.
        # must be defined here as a property for mypy
        raise AbstractMethodError(self)

    @property
    def _values(self) -> ExtensionArray | np.ndarray:
        # [FUNCTION DOC]
        # Abstract property declared for type checkers; raises the
        # abstract-method error. Subclasses return the underlying array.
        # must be defined here as a property for mypy
        raise AbstractMethodError(self)

    @final
    def transpose(self, *args, **kwargs) -> Self:
        # [FUNCTION DOC]
        # Transpose, which is the object itself for one dimension. Extra
        # arguments are only validated (numpy compatibility: anything other than
        # default values raises). Also exposed as the T property. Returns self.
        """
        Return the transpose, which is by definition self.

        Returns
        -------
        %(klass)s
        """
        nv.validate_transpose(args, kwargs)
        return self

    T = property(
        transpose,
        doc="""
        Return the transpose, which is by definition self.

        See Also
        --------
        Index : Immutable sequence used for indexing and alignment.

        Examples
        --------
        For Series:

        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: str
        >>> s.T
        0     Ant
        1    Bear
        2     Cow
        dtype: str

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx.T
        Index([1, 2, 3], dtype='int64')
        """,
    )

    @property
    def shape(self) -> Shape:
        # [FUNCTION DOC]
        # Property: the shape tuple of the underlying values array.
        """
        Return a tuple of the shape of the underlying data.

        See Also
        --------
        Series.ndim : Number of dimensions of the underlying data.
        Series.size : Return the number of elements in the underlying data.
        Series.nbytes : Return the number of bytes in the underlying data.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.shape
        (3,)
        """
        return self._values.shape

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Abstract length declared for type checkers; raises the abstract-method
        # error. Subclasses return the number of elements.
        # We need this defined here for mypy
        raise AbstractMethodError(self)

    # Temporarily avoid using `-> Literal[1]:` because of an IPython (jedi) bug
    # https://github.com/ipython/ipython/issues/14412
    # https://github.com/davidhalter/jedi/issues/1990
    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Property: always 1.
        """
        Number of dimensions of the underlying data, by definition 1.

        See Also
        --------
        Series.size: Return the number of elements in the underlying data.
        Series.shape: Return a tuple of the shape of the underlying data.
        Series.dtype: Return the dtype object of the underlying data.
        Series.values: Return Series as ndarray or ndarray-like depending on the dtype.

        Examples
        --------
        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: str
        >>> s.ndim
        1

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.ndim
        1
        """
        return 1

    @final
    def item(self):
        # [FUNCTION DOC]
        # Returns the sole element as a Python or pandas scalar (the first item of
        # the iterator). Raises a value error if the length is not exactly one.
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of Series or Index.

        Raises
        ------
        ValueError
            If the data is not length = 1.

        See Also
        --------
        Index.values : Returns an array representing the data in the Index.
        Series.head : Returns the first `n` rows.

        Examples
        --------
        >>> s = pd.Series([1])
        >>> s.item()
        1

        For an index:

        >>> s = pd.Series([1], index=["a"])
        >>> s.index.item()
        'a'
        """
        if len(self) == 1:
            return next(iter(self))
        raise ValueError("can only convert an array of size 1 to a Python scalar")

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Property: byte size of the underlying values array.
        """
        Return the number of bytes in the underlying data.

        See Also
        --------
        Series.ndim : Number of dimensions of the underlying data.
        Series.size : Return the number of elements in the underlying data.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: str
        >>> s.nbytes
        34

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.nbytes
        24
        """
        return self._values.nbytes

    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property: number of elements, taken as the length of the values array.
        """
        Return the number of elements in the underlying data.

        See Also
        --------
        Series.ndim: Number of dimensions of the underlying data, by definition 1.
        Series.shape: Return a tuple of the shape of the underlying data.
        Series.dtype: Return the dtype object of the underlying data.
        Series.values: Return Series as ndarray or ndarray-like depending on the dtype.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: str
        >>> s.size
        3

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.size
        3
        """
        return len(self._values)

    @property
    def array(self) -> ExtensionArray:
        # [FUNCTION DOC]
        # Abstract property: the extension array backing the object (numpy data
        # wrapped without copying). This base version raises the abstract-method
        # error; subclasses implement it.
        """
        The ExtensionArray of the data backing this Series or Index.

        This property provides direct access to the underlying array data of a
        Series or Index without requiring conversion to a NumPy array. It
        returns an ExtensionArray, which is the native storage format for
        pandas extension dtypes.

        Returns
        -------
        ExtensionArray
            An ExtensionArray of the values stored within. For extension
            types, this is the actual array. For NumPy native types, this
            is a thin (no copy) wrapper around :class:`numpy.ndarray`.

            ``.array`` differs from ``.values``, which may require converting
            the data to a different form.

        See Also
        --------
        Index.to_numpy : Similar method that always returns a NumPy array.
        Series.to_numpy : Similar method that always returns a NumPy array.

        Notes
        -----
        This table lays out the different array types for each extension
        dtype within pandas.

        ================== =============================
        dtype              array type
        ================== =============================
        category           Categorical
        period             PeriodArray
        interval           IntervalArray
        IntegerNA          IntegerArray
        string             StringArray
        boolean            BooleanArray
        datetime64[ns, tz] DatetimeArray
        ================== =============================

        For any 3rd-party extension types, the array type will be an
        ExtensionArray.

        For all remaining dtypes ``.array`` will be a
        :class:`arrays.NumpyExtensionArray` wrapping the actual ndarray
        stored within. If you absolutely need a NumPy array (possibly with
        copying / coercing data), then use :meth:`Series.to_numpy` instead.

        Examples
        --------
        For regular NumPy types like int, and float, a NumpyExtensionArray
        is returned.

        >>> pd.Series([1, 2, 3]).array
        <NumpyExtensionArray>
        [1, 2, 3]
        Length: 3, dtype: int64

        For extension types, like Categorical, the actual ExtensionArray
        is returned

        >>> ser = pd.Series(pd.Categorical(["a", "b", "a"]))
        >>> ser.array
        ['a', 'b', 'a']
        Categories (2, str): ['a', 'b']
        """
        raise AbstractMethodError(self)

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
        **kwargs,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Converts the values to a numpy array. Inputs: dtype (default none), the
        # target dtype; copy (default false); na_value (default: not given), the
        # replacement for missing values; kwargs, extra options accepted only for
        # extension dtypes.
        # Extension dtypes delegate to the backing array's conversion with all
        # arguments. For numpy dtypes, extra keyword arguments raise a type error
        # naming the first one. If na_value is given (except NaN on a float dtype)
        # and there are missing values, they are replaced in a copy (a copy is
        # avoided when the array cannot hold the value because the dtype
        # conversion already copies). The values are converted with the requested
        # dtype. Finally, when the result shares memory with the object's data:
        # with copy false it is returned as a read-only view; with copy true and no
        # replacement made it is copied. Returns the array.
        # KNOWN ISSUE: the docstring's notes say that for numpy dtypes the result
        # is a reference to the data that can be modified in place, but with the
        # default of no copy the code returns a read-only view, so in-place writes
        # raise an error.
        """
        A NumPy ndarray representing the values in this Series or Index.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.
        **kwargs
            Additional keywords passed through to the ``to_numpy`` method
            of the underlying array (for extension arrays).

        Returns
        -------
        numpy.ndarray
            The NumPy ndarray holding the values from this Series or Index.
            The dtype of the array may differ. See Notes.

        See Also
        --------
        Series.array : Get the actual data stored within.
        Index.array : Get the actual data stored within.
        DataFrame.to_numpy : Similar method for DataFrame.

        Notes
        -----
        The returned array will be the same up to equality (values equal
        in `self` will be equal in the returned array; likewise for values
        that are not equal). When `self` contains an ExtensionArray, the
        dtype may be different. For example, for a category-dtype Series,
        ``to_numpy()`` will return a NumPy array and the categorical dtype
        will be lost.

        For NumPy dtypes, this will be a reference to the actual data stored
        in this Series or Index (assuming ``copy=False``). Modifying the result
        in place will modify the data stored in the Series or Index (not that
        we recommend doing that).

        For extension types, ``to_numpy()`` *may* require copying data and
        coercing the result to a NumPy type (possibly object), which may be
        expensive. When you need a no-copy reference to the underlying data,
        :attr:`Series.array` should be used instead.

        This table lays out the different dtypes and default return types of
        ``to_numpy()`` for various dtypes within pandas.

        ================== ================================
        dtype              array type
        ================== ================================
        category[T]        ndarray[T] (same dtype as input)
        period             ndarray[object] (Periods)
        interval           ndarray[object] (Intervals)
        IntegerNA          ndarray[object]
        datetime64[ns]     datetime64[ns]
        datetime64[ns, tz] ndarray[object] (Timestamps)
        ================== ================================

        Examples
        --------
        >>> ser = pd.Series(pd.Categorical(["a", "b", "a"]))
        >>> ser.to_numpy()
        array(['a', 'b', 'a'], dtype=object)

        Specify the `dtype` to control how datetime-aware data is represented.
        Use ``dtype=object`` to return an ndarray of pandas :class:`Timestamp`
        objects, each with the correct ``tz``.

        >>> ser = pd.Series(pd.date_range("2000", periods=2, tz="CET"))
        >>> ser.to_numpy(dtype=object)
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or ``dtype='datetime64[ns]'`` to return an ndarray of native
        datetime64 values. The values are converted to UTC and the timezone
        info is dropped.

        >>> ser.to_numpy(dtype="datetime64[ns]")
        ... # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00...'],
              dtype='datetime64[ns]')
        """
        if isinstance(self.dtype, ExtensionDtype):
            return self.array.to_numpy(dtype, copy=copy, na_value=na_value, **kwargs)
        elif kwargs:
            bad_keys = next(iter(kwargs.keys()))
            raise TypeError(
                f"to_numpy() got an unexpected keyword argument '{bad_keys}'"
            )

        fillna = (
            na_value is not lib.no_default
            # no need to fillna with np.nan if we already have a float dtype
            and not (na_value is np.nan and np.issubdtype(self.dtype, np.floating))
        )

        values = self._values
        if fillna and self.hasnans:
            if not can_hold_element(values, na_value):
                # if we can't hold the na_value asarray either makes a copy or we
                # error before modifying values. The asarray later on thus won't make
                # another copy
                values = np.asarray(values, dtype=dtype)
            else:
                values = values.copy()

            values[np.asanyarray(isna(self))] = na_value

        result = np.asarray(values, dtype=dtype)

        if (copy and not fillna) or not copy:
            if np.shares_memory(self._values[:2], result[:2]):
                # Take slices to improve performance of check
                if not copy:
                    result = result.view()
                    result.flags.writeable = False
                else:
                    result = result.copy()

        return result

    @final
    @property
    def empty(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the object has no elements (size is zero).
        """
        Indicator whether Index is empty.

        An Index is considered empty if it has no elements. This property can be
        useful for quickly checking the state of an Index, especially in data
        processing and analysis workflows where handling of empty datasets might
        be required.

        Returns
        -------
        bool
            If Index is empty, return True, if not return False.

        See Also
        --------
        Index.size : Return the number of elements in the underlying data.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.empty
        False

        >>> idx_empty = pd.Index([])
        >>> idx_empty
        Index([], dtype='object')
        >>> idx_empty.empty
        True

        If we only have NaNs in our DataFrame, it is not considered empty!

        >>> idx = pd.Index([np.nan, np.nan])
        >>> idx
        Index([nan, nan], dtype='float64')
        >>> idx.empty
        False
        """
        return not self.size

    def argmax(
        self, axis: AxisInt | None = None, skipna: bool = True, *args, **kwargs
    ) -> int:
        # [FUNCTION DOC]
        # Position of the first maximum. Inputs: axis (ignored; validated to be
        # none or a valid axis for compatibility); skipna (default true);
        # args/kwargs, numpy compatibility arguments that are validated (and may
        # carry a skipna override check). Extension-array values delegate to the
        # array's own method; numpy values use the nan-aware routine. Per the
        # documentation, a ValueError arises when all values are missing or
        # skipna is false with a missing value present. Returns the integer
        # position (numpy integer for numpy data).
        """
        Return int position of the largest value in the Series.

        If the maximum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : None
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the maximum value.

        See Also
        --------
        Series.argmax : Return position of the maximum value.
        Series.argmin : Return position of the minimum value.
        numpy.ndarray.argmax : Equivalent method for numpy arrays.
        Series.idxmax : Return index label of the maximum values.
        Series.idxmin : Return index label of the minimum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series(
        ...     [100.0, 110.0, 120.0, 110.0],
        ...     index=[
        ...         "Corn Flakes",
        ...         "Almond Delight",
        ...         "Cinnamon Toast Crunch",
        ...         "Cocoa Puff",
        ...     ],
        ... )
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        np.int64(2)
        >>> s.argmin()
        np.int64(0)

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)

        if isinstance(delegate, ExtensionArray):
            return delegate.argmax(skipna=skipna)
        else:
            result = nanops.nanargmax(delegate, skipna=skipna)
            # error: Incompatible return value type (got "Union[int, ndarray]", expected
            # "int")
            return result  # type: ignore[return-value]

    def argmin(
        self, axis: AxisInt | None = None, skipna: bool = True, *args, **kwargs
    ) -> int:
        # [FUNCTION DOC]
        # Position of the first minimum; mirrors the maximum-position method in
        # inputs, delegation and error behaviour, using the minimum variants.
        """
        Return int position of the smallest value in the Series.

        If the minimum is achieved in multiple locations,
        the first row position is returned.

        Parameters
        ----------
        axis : None
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        int
            Row position of the minimum value.

        See Also
        --------
        Series.argmin : Return position of the minimum value.
        Series.argmax : Return position of the maximum value.
        numpy.ndarray.argmin : Equivalent method for numpy arrays.
        Series.idxmin : Return index label of the minimum values.
        Series.idxmax : Return index label of the maximum values.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = pd.Series(
        ...     [100.0, 110.0, 120.0, 110.0],
        ...     index=[
        ...         "Corn Flakes",
        ...         "Almond Delight",
        ...         "Cinnamon Toast Crunch",
        ...         "Cocoa Puff",
        ...     ],
        ... )
        >>> s
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()
        np.int64(2)
        >>> s.argmin()
        np.int64(0)

        The maximum cereal calories is the third element and
        the minimum cereal calories is the first element,
        since series is zero-indexed.
        """
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)

        if isinstance(delegate, ExtensionArray):
            return delegate.argmin(skipna=skipna)
        else:
            result = nanops.nanargmin(delegate, skipna=skipna)
            # error: Incompatible return value type (got "Union[int, ndarray]", expected
            # "int")
            return result  # type: ignore[return-value]

    def tolist(self) -> list:
        # [FUNCTION DOC]
        # Returns the values as a Python list of scalars (Python scalars for
        # basic types, pandas scalars for datetime-like and interval/period
        # types) via the values array's own list conversion. Alias: to_list.
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list
            List containing the values as Python or pandas scalers.

        See Also
        --------
        numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
            nested list of Python scalars.

        Examples
        --------
        For Series

        >>> s = pd.Series([1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        >>> idx.to_list()
        [1, 2, 3]
        """
        return self._values.tolist()

    to_list = tolist

    def __iter__(self) -> Iterator:
        # [FUNCTION DOC]
        # Returns an iterator over scalar elements. For non-numpy values (extension
        # and datetime-like arrays) iterates the array directly; for numpy arrays
        # lazily maps the array's scalar extraction over positions so elements are
        # Python scalars.
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        iterator
            An iterator yielding scalar values from the Series.

        See Also
        --------
        Series.items : Lazily iterate over (index, value) tuples.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> for x in s:
        ...     print(x)
        1
        2
        3
        """
        # We are explicitly making element iterators.
        if not isinstance(self._values, np.ndarray):
            # Check type instead of dtype to catch DTA/TDA
            return iter(self._values)
        else:
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: True if any element is missing (using the library's
        # missing-value test). Cached, so only valid while the data is unchanged.
        """
        Return True if there are any NaNs.

        Enables various performance speedups.

        Returns
        -------
        bool

        See Also
        --------
        Series.isna : Detect missing values.
        Series.notna : Detect existing (non-missing) values.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, None])
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        dtype: float64
        >>> s.hasnans
        True
        """
        # error: Item "bool" of "Union[bool, ndarray[Any, dtype[bool_]], NDFrame]"
        # has no attribute "any"
        return bool(isna(self).any())  # type: ignore[union-attr]

    @final
    def _map_values(self, mapper, na_action=None):
        # [FUNCTION DOC]
        # HELPER: maps the values through a mapping. Inputs: mapper, a function,
        # dict or Series; na_action (default none), "ignore" to leave missing
        # values out of the mapping. Extension arrays use their own map method;
        # numpy arrays use the generic map routine. Returns the mapped array.
        # KNOWN ISSUE: the docstring says the result is an Index or MultiIndex,
        # but the method returns the mapped array (numpy or extension) and leaves
        # any wrapping to callers.
        """
        An internal function that maps values using the input
        correspondence (which can be a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            The input correspondence object
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping function

        Returns
        -------
        Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        arr = self._values

        if isinstance(arr, ExtensionArray):
            return arr.map(mapper, na_action=na_action)

        return algorithms.map_array(arr, mapper, na_action=na_action)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series:
        # [FUNCTION DOC]
        # Counts occurrences of unique values and returns them as a Series.
        # Inputs: normalize (default false), return fractions; sort (default
        # true), order by count (stable) else keep order of appearance; ascending
        # (default false); bins (default none), count numeric values in this many
        # half-open bins; dropna (default true), exclude missing values. The
        # work is delegated to the shared algorithms routine with these options.
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : bool, default True
            Stable sort by frequencies when True. Preserve the order of the data
            when False.

            .. versionchanged:: 3.0.0

                Prior to 3.0.0, the sort was unstable.
        ascending : bool, default False
            Sort in ascending order.
        bins : int, optional
            Rather than count values, group them into half-open bins,
            a convenience for ``pd.cut``, only works with numeric data.
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        Series
            Series containing counts of unique values.

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.count: Number of non-NA elements in a DataFrame.
        DataFrame.value_counts: Equivalent method on DataFrames.

        Examples
        --------
        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
        >>> index.value_counts()
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        Name: count, dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(normalize=True)
        3.0    0.4
        1.0    0.2
        2.0    0.2
        4.0    0.2
        Name: proportion, dtype: float64

        **bins**

        Bins can be useful for going from a continuous variable to a
        categorical variable; instead of counting unique
        apparitions of values, divide the index in the specified
        number of half-open bins.

        >>> s.value_counts(bins=3)
        (0.996, 2.0]    2
        (2.0, 3.0]      2
        (3.0, 4.0]      1
        Name: count, dtype: int64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> s.value_counts(dropna=False)
        3.0    2
        1.0    1
        2.0    1
        4.0    1
        NaN    1
        Name: count, dtype: int64

        **Categorical Dtypes**

        Rows with categorical type will be counted as one group
        if they have same categories and order.
        In the example below, even though ``a``, ``c``, and ``d``
        all have the same data types of ``category``,
        only ``c`` and ``d`` will be counted as one group
        since ``a`` doesn't have the same categories.

        >>> df = pd.DataFrame({"a": [1], "b": ["2"], "c": [3], "d": [3]})
        >>> df = df.astype({"a": "category", "c": "category", "d": "category"})
        >>> df
           a  b  c  d
        0  1  2  3  3

        >>> df.dtypes
        a    category
        b      str
        c    category
        d    category
        dtype: object

        >>> df.dtypes.value_counts()
        category    2
        category    1
        str         1
        Name: count, dtype: int64
        """
        return algorithms.value_counts_internal(
            self,
            sort=sort,
            ascending=ascending,
            normalize=normalize,
            bins=bins,
            dropna=dropna,
        )

    def unique(self):
        # [FUNCTION DOC]
        # Returns the unique values in order of first appearance. Extension arrays
        # use their own method (returning an extension array); numpy arrays use the
        # generic one-dimensional routine (returning a numpy array).
        values = self._values
        if not isinstance(values, np.ndarray):
            # i.e. ExtensionArray
            result = values.unique()
        else:
            result = algorithms.unique1d(values)  # type: ignore[assignment]
        return result

    @final
    def nunique(self, dropna: bool = True) -> int:
        # [FUNCTION DOC]
        # Number of distinct values. Input dropna (default true): exclude missing
        # values from the count. Computed as the length of the unique values after
        # optionally removing missing entries.
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the count.

        Returns
        -------
        int
            An integer indicating the number of unique elements in the object.

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> s = pd.Series([1, 3, 5, 7, 7])
        >>> s
        0    1
        1    3
        2    5
        3    7
        4    7
        dtype: int64

        >>> s.nunique()
        4
        """
        uniqs = self.unique()
        if dropna:
            uniqs = remove_na_arraylike(uniqs)
        return len(uniqs)

    @property
    def is_unique(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when no value repeats (missing values count as a value):
        # the distinct count including missing equals the length.
        """
        Return True if values in the object are unique.

        Returns
        -------
        bool

        See Also
        --------
        Series.unique : Return unique values of Series object.
        Series.drop_duplicates : Return Series with duplicate values removed.
        Series.duplicated : Indicate duplicate Series values.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.is_unique
        True

        >>> s = pd.Series([1, 2, 3, 1])
        >>> s.is_unique
        False
        """
        return self.nunique(dropna=False) == len(self)

    @property
    def is_monotonic_increasing(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if values are non-decreasing; computed by building an
        # Index from the object and asking it (imported lazily).
        """
        Return True if values in the object are monotonically increasing.

        Returns
        -------
        bool

        See Also
        --------
        Series.is_monotonic_decreasing : Return boolean if values in the object are
            monotonically decreasing.

        Examples
        --------
        >>> s = pd.Series([1, 2, 2])
        >>> s.is_monotonic_increasing
        True

        >>> s = pd.Series([3, 2, 1])
        >>> s.is_monotonic_increasing
        False
        """
        from pandas import Index

        return Index(self).is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if values are non-increasing; computed through an Index
        # built from the object.
        """
        Return True if values in the object are monotonically decreasing.

        Returns
        -------
        bool

        See Also
        --------
        Series.is_monotonic_increasing : Return boolean if values in the object are
            monotonically increasing.

        Examples
        --------
        >>> s = pd.Series([3, 2, 2, 1])
        >>> s.is_monotonic_decreasing
        True

        >>> s = pd.Series([1, 2, 3])
        >>> s.is_monotonic_decreasing
        False
        """
        from pandas import Index

        return Index(self).is_monotonic_decreasing

    @final
    def _memory_usage(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # HELPER: memory use of the values in bytes. Input deep (default false):
        # also measure the Python objects referenced by object-dtype data (skipped
        # on PyPy). If the backing array provides its own memory-usage method it is
        # used; otherwise the array's byte size plus, for deep object data, the
        # objects' sizes. Returns an integer.
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
        if hasattr(self.array, "memory_usage"):
            return self.array.memory_usage(  # pyright: ignore[reportAttributeAccessIssue]
                deep=deep,
            )

        v = self.array.nbytes
        if deep and is_object_dtype(self.dtype) and not PYPY:
            values = cast(np.ndarray, self._values)
            v += lib.memory_usage_of_objects(values)
        return v

    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> tuple[npt.NDArray[np.intp], Index]:
        # [FUNCTION DOC]
        # Encodes values as integer codes plus the array of unique values. Inputs:
        # sort (default false): sort the uniques and remap codes; use_na_sentinel
        # (default true): code missing values as -1 and omit them from the uniques
        # rather than giving them their own code. Uses the shared factorization
        # routine on the underlying values; half-precision float uniques are
        # widened to single precision. The uniques are returned as a multi-level
        # index for multi-level indexes (empty slice for an empty one, else rebuilt
        # with the same class) and as an Index with the original dtype otherwise
        # (falling back to an inferred dtype if the dtype is unsupported by
        # Index). Returns (codes, uniques).
        """
        Encode the object as an enumerated type or categorical variable.

        This method is useful for obtaining a numeric representation of an
        array when all that matters is identifying distinct values. `factorize`
        is available as both a top-level function :func:`pandas.factorize`,
        and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

        Parameters
        ----------
        sort : bool, default False
            Sort `uniques` and shuffle `codes` to maintain the
            relationship.
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

        Returns
        -------
        codes : ndarray
            An integer ndarray that's an indexer into `uniques`.
            ``uniques.take(codes)`` will have the same values as `values`.
        uniques : ndarray, Index, or Categorical
            The unique valid values. When `values` is Categorical, `uniques`
            is a Categorical. When `values` is some other pandas object, an
            `Index` is returned. Otherwise, a 1-D ndarray is returned.

            .. note::

                Even if there's a missing value in `values`, `uniques` will
                *not* contain an entry for it.

        See Also
        --------
        cut : Discretize continuous-valued array.
        unique : Find the unique value in an array.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.factorize>` for more examples.

        Examples
        --------
        These examples all show factorize as a top-level method like
        ``pd.factorize(values)``. The results are identical for methods like
        :meth:`Series.factorize`.

        >>> codes, uniques = pd.factorize(
        ...     np.array(["b", "b", "a", "c", "b"], dtype="O")
        ... )
        >>> codes
        array([0, 0, 1, 2, 0])
        >>> uniques
        array(['b', 'a', 'c'], dtype=object)

        With ``sort=True``, the `uniques` will be sorted, and `codes` will be
        shuffled so that the relationship is the maintained.

        >>> codes, uniques = pd.factorize(
        ...     np.array(["b", "b", "a", "c", "b"], dtype="O"), sort=True
        ... )
        >>> codes
        array([1, 1, 0, 2, 1])
        >>> uniques
        array(['a', 'b', 'c'], dtype=object)

        When ``use_na_sentinel=True`` (the default), missing values are indicated in
        the `codes` with the sentinel value ``-1`` and missing values are not
        included in `uniques`.

        >>> codes, uniques = pd.factorize(
        ...     np.array(["b", None, "a", "c", "b"], dtype="O")
        ... )
        >>> codes
        array([ 0, -1,  1,  2,  0])
        >>> uniques
        array(['b', 'a', 'c'], dtype=object)

        Thus far, we've only factorized lists (which are internally coerced to
        NumPy arrays). When factorizing pandas objects, the type of `uniques`
        will differ. For Categoricals, a `Categorical` is returned.

        >>> cat = pd.Categorical(["a", "a", "c"], categories=["a", "b", "c"])
        >>> codes, uniques = pd.factorize(cat)
        >>> codes
        array([0, 0, 1])
        >>> uniques
        ['a', 'c']
        Categories (3, str): ['a', 'b', 'c']

        Notice that ``'b'`` is in ``uniques.categories``, despite not being
        present in ``cat.values``.

        For all other pandas objects, an Index of the appropriate type is
        returned.

        >>> cat = pd.Series(["a", "a", "c"])
        >>> codes, uniques = pd.factorize(cat)
        >>> codes
        array([0, 0, 1])
        >>> uniques
        Index(['a', 'c'], dtype='str')

        If NaN is in the values, and we want to include NaN in the uniques of the
        values, it can be achieved by setting ``use_na_sentinel=False``.

        >>> values = np.array([1, 2, 1, np.nan])
        >>> codes, uniques = pd.factorize(values)  # default: use_na_sentinel=True
        >>> codes
        array([ 0,  1,  0, -1])
        >>> uniques
        array([1., 2.])

        >>> codes, uniques = pd.factorize(values, use_na_sentinel=False)
        >>> codes
        array([0, 1, 0, 2])
        >>> uniques
        array([ 1.,  2., nan])
        """
        codes, uniques = algorithms.factorize(
            self._values, sort=sort, use_na_sentinel=use_na_sentinel
        )
        if uniques.dtype == np.float16:
            uniques = uniques.astype(np.float32)

        if isinstance(self, ABCMultiIndex):
            # preserve MultiIndex
            if len(self) == 0:
                # GH#57517
                uniques = self[:0]
            else:
                uniques = self._constructor(uniques)
        else:
            from pandas import Index

            try:
                uniques = Index(uniques, dtype=self.dtype, copy=False)
            except NotImplementedError:
                # not all dtypes are supported in Index that are allowed for Series
                # e.g. float16 or bytes
                uniques = Index(uniques, copy=False)
        return codes, uniques

    # This overload is needed so that the call to searchsorted in
    # pandas.core.resample.TimeGrouper._get_period_bins picks the correct result

    # error: Overloaded function signatures 1 and 2 overlap with incompatible
    # return types
    @overload
    def searchsorted(  # type: ignore[overload-overlap]
        self,
        value: ScalarLike_co,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> np.intp:
        # [FUNCTION DOC]
        # Typing overload stub (scalar value gives one position); see the
        # implementation below.
        ...

    @overload
    def searchsorted(
        self,
        value: npt.ArrayLike | ExtensionArray,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter = ...,
    ) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Typing overload stub (array value gives an array of positions); see the
        # implementation below.
        ...

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Binary search for insertion positions in a sorted object. Inputs: value,
        # a scalar or array-like; side, "left" (default) for the first suitable
        # position or "right" for the last; sorter (default none), integer indices
        # that sort the object. A data frame value raises a value error. Extension
        # arrays use their own method directly; numpy values use the generic
        # routine. The sortedness of the object is not checked. Returns a single
        # position for scalar input or an array of positions.
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted Index `self` such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        .. note::

            The Index *must* be monotonically sorted, otherwise
            wrong locations will likely be returned. Pandas does *not*
            check this for you.

        Parameters
        ----------
        value : array-like or scalar
            Values to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort `self` into ascending
            order. They are typically the result of ``np.argsort``.

        Returns
        -------
        int or array of int
            A scalar or array of insertion points with the
            same shape as `value`.

        See Also
        --------
        sort_values : Sort by the values along either axis.
        numpy.searchsorted : Similar method from NumPy.

        Notes
        -----
        Binary search is used to find the required insertion points.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> ser
        0    1
        1    2
        2    3
        dtype: int64

        >>> ser.searchsorted(4)
        np.int64(3)

        >>> ser.searchsorted([0, 4])
        array([0, 3])

        >>> ser.searchsorted([1, 3], side="left")
        array([0, 2])

        >>> ser.searchsorted([1, 3], side="right")
        array([1, 3])

        >>> ser = pd.Series(pd.to_datetime(["3/11/2000", "3/12/2000", "3/13/2000"]))
        >>> ser
        0   2000-03-11
        1   2000-03-12
        2   2000-03-13
        dtype: datetime64[us]

        >>> ser.searchsorted("3/14/2000")
        np.int64(3)

        >>> ser = pd.Categorical(
        ...     ["apple", "bread", "bread", "cheese", "milk"], ordered=True
        ... )
        >>> ser
        ['apple', 'bread', 'bread', 'cheese', 'milk']
        Categories (4, str): ['apple' < 'bread' < 'cheese' < 'milk']

        >>> ser.searchsorted("bread")
        np.int64(1)

        >>> ser.searchsorted(["bread"], side="right")
        array([3])

        If the values are not monotonically sorted, wrong locations
        may be returned:

        >>> ser = pd.Series([2, 1, 3])
        >>> ser
        0    2
        1    1
        2    3
        dtype: int64

        >>> ser.searchsorted(1)  # doctest: +SKIP
        0  # wrong result, correct would be 1
        """
        if isinstance(value, ABCDataFrame):
            msg = (
                "Value must be 1-D array-like or scalar, "
                f"{type(value).__name__} is not supported"
            )
            raise ValueError(msg)

        values = self._values
        if not isinstance(values, np.ndarray):
            # Going through EA.searchsorted directly improves performance GH#38083
            return values.searchsorted(value, side=side, sorter=sorter)

        return algorithms.searchsorted(
            values,
            value,
            side=side,
            sorter=sorter,
        )

    def drop_duplicates(self, *, keep: DropKeep = "first") -> Self:
        # [FUNCTION DOC]
        # Returns the object without duplicate values. Input keep (keyword-only,
        # default "first"): which duplicate to retain (first, last, or none of
        # them). Builds the duplicate mask and returns the object indexed by its
        # negation.
        duplicated = self._duplicated(keep=keep)
        # error: Value of type "IndexOpsMixin" is not indexable
        return self[~duplicated]  # type: ignore[index]

    @final
    def _duplicated(self, keep: DropKeep = "first") -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # HELPER: boolean mask marking duplicate values. Input keep (default
        # "first"): which occurrence is not marked (first, last, or none kept so
        # all duplicates are marked). Extension arrays use their own method, numpy
        # arrays the generic routine.
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.duplicated(keep=keep)
        return algorithms.duplicated(arr, keep=keep)

    def _arith_method(self, other, op):
        # [FUNCTION DOC]
        # HELPER: implements a binary arithmetic operator. Inputs: other, the other
        # operand; op, the operator function. The result name is derived from the
        # two operands' names. The other operand is unboxed to a raw array (ranges
        # become integer arrays, scalars are prepared for the operation, and
        # datetime-like numpy arrays are wrapped). The operation runs with numpy
        # floating-point warnings suppressed, then the subclass's result builder
        # wraps the output with the chosen name.
        res_name = ops.get_op_result_name(self, other)

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        rvalues = ops.maybe_prepare_scalar_for_op(rvalues, lvalues.shape)
        rvalues = ensure_wrapped_if_datetimelike(rvalues)
        if isinstance(rvalues, range):
            rvalues = np.arange(rvalues.start, rvalues.stop, rvalues.step)

        with np.errstate(all="ignore"):
            result = ops.arithmetic_op(lvalues, rvalues, op)

        return self._construct_result(result, name=res_name, other=other)

    def _construct_result(self, result, name, other):
        # [FUNCTION DOC]
        # HELPER: abstract hook that wraps a raw array result of an arithmetic
        # operation into the proper container. Inputs: result, name, other (the
        # other operand). Always raises the abstract-method error here.
        """
        Construct an appropriately-wrapped result from the ArrayLike result
        of an arithmetic-like operation.
        """
        raise AbstractMethodError(self)
