"""
An interface for extending pandas with custom arrays.

.. warning::

   This is an experimental API and subject to breaking changes
   without warning.
"""

# [MODULE DOC]
# Defines the public extension array interface that third parties subclass to add
# custom array types, and the helper mixins used to implement operators.
# Public contents: the abstract extension array base class (with default
# implementations of most methods built on a few required ones), a subclass that
# declares any/all support for typing, a mixin that attaches operator methods to a
# class by name, and a public mixin that implements operators by applying them to
# scalar elements. It also holds a module-level dictionary of shared docstring
# templates (currently one entry, for repeat).
# Relationships: concrete arrays elsewhere in the arrays package (masked, datetime-like,
# categorical, arrow, string, etc.) inherit from the base class; the dtype counterpart
# lives in the core dtypes package. The groupby hook imports the groupby and string
# modules lazily to avoid import cycles.
# Import-time side effects: none beyond creating the shared-docs dictionary and class
# definitions (the public classes are re-labelled as belonging to the extensions
# namespace).
from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_insert_loc,
)

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    arraylike,
    missing,
    roperator,
)
from pandas.core.algorithms import (
    duplicated,
    factorize_array,
    isin,
    map_array,
    mode,
    rank,
    unique,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import (
    nargminmax,
    nargsort,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

    from pandas._libs.missing import NAType
    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        AxisInt,
        Dtype,
        DtypeObj,
        FillnaOptions,
        InterpolateOptions,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        ScalarIndexer,
        SequenceIndexer,
        Shape,
        SortKind,
        TakeIndexer,
        npt,
    )

    from pandas import (
        Index,
        Series,
    )

_extension_array_shared_docs: dict[str, str] = {}


@set_module("pandas.api.extensions")
class ExtensionArray:
    # [CLASS DOC]
    # Abstract base class for custom one-dimensional array types that pandas treats as
    # first-class arrays (public in the extensions API namespace). It deliberately does
    # not use an abstract-base metaclass for speed; required members raise an abstract-
    # method error instead, and there is no virtual-subclass registration. Class state:
    # a type tag identifying extension arrays to the generic-type checks (must not be
    # overridden), a priority value that orders it after index, series and dataframe in
    # binary operations, and a read-only flag (default false) that mutating methods
    # consult. Instances are unhashable by default. Required interface (subclass must
    # implement): the sequence constructors, the factorized reconstruction hook, item
    # access, length, equality, dtype, byte size, missing-value detection, take, copy
    # and concatenation (the interpolate method raises not-implemented by default).
    # Everything else has a generic default implementation, mostly built from the
    # required methods by converting to object dtype (slow; subclasses are encouraged to
    # override fill, pad, dropna, unique, factorize, argsort/argmin/argmax, searchsorted
    # and map). Optional hooks: accumulate, reduce, string-sequence construction,
    # hashing, formatting, explode and the groupby operation hook. Items are limited to
    # one dimension, so shape, size, transpose and ravel are trivial. If the subclass
    # implements the numpy ufunc hook it must return the not-implemented sentinel when
    # any series is among the inputs and define a tuple of handled types.
    """
    Abstract base class for custom 1-D array types.

    pandas will recognize instances of this class as proper arrays
    with a custom type and will not attempt to coerce them to objects. They
    may be stored directly inside a :class:`DataFrame` or :class:`Series`.

    Attributes
    ----------
    dtype
    nbytes
    ndim
    shape

    Methods
    -------
    argsort
    astype
    copy
    dropna
    duplicated
    factorize
    fillna
    equals
    insert
    interpolate
    isin
    isna
    item
    ravel
    repeat
    searchsorted
    shift
    take
    tolist
    unique
    view
    _accumulate
    _concat_same_type
    _explode
    _formatter
    _from_factorized
    _from_sequence
    _from_sequence_of_strings
    _hash_pandas_object
    _pad_or_backfill
    _reduce
    _values_for_argsort
    _values_for_factorize

    See Also
    --------
    api.extensions.ExtensionDtype : A custom data type, to be paired with an
        ExtensionArray.
    api.extensions.ExtensionArray.dtype : An instance of ExtensionDtype.

    Notes
    -----
    The interface includes the following abstract methods that must be
    implemented by subclasses:

    * _from_sequence
    * _from_factorized
    * __getitem__
    * __len__
    * __eq__
    * dtype
    * nbytes
    * isna
    * take
    * copy
    * _concat_same_type
    * interpolate

    A default repr displaying the type, (truncated) data, length,
    and dtype is provided. It can be customized or replaced by
    by overriding:

    * __repr__ : A default repr for the ExtensionArray.
    * _formatter : Print scalars inside a Series or DataFrame.

    Some methods require casting the ExtensionArray to an ndarray of Python
    objects with ``self.astype(object)``, which may be expensive. When
    performance is a concern, we highly recommend overriding the following
    methods:

    * fillna
    * _pad_or_backfill
    * dropna
    * unique
    * factorize / _values_for_factorize
    * argsort, argmax, argmin / _values_for_argsort
    * searchsorted
    * map

    The remaining methods implemented on this class should be performant,
    as they only compose abstract methods. Still, a more efficient
    implementation may be available, and these methods can be overridden.

    One can implement methods to handle array accumulations or reductions.

    * _accumulate
    * _reduce

    One can implement methods to handle parsing from strings that will be used
    in methods such as ``pandas.io.parsers.read_csv``.

    * _from_sequence_of_strings

    This class does not inherit from 'abc.ABCMeta' for performance reasons.
    Methods and properties required by the interface raise
    ``pandas.errors.AbstractMethodError`` and no ``register`` method is
    provided for registering virtual subclasses.

    ExtensionArrays are limited to 1 dimension.

    They may be backed by none, one, or many NumPy arrays. For example,
    ``pandas.Categorical`` is an extension array backed by two arrays,
    one for codes and one for categories. An array of IPv6 address may
    be backed by a NumPy structured array with two fields, one for the
    lower 64 bits and one for the upper 64 bits. Or they may be backed
    by some other storage type, like Python lists. Pandas makes no
    assumptions on how the data are stored, just that it can be converted
    to a NumPy array.
    The ExtensionArray interface does not impose any rules on how this data
    is stored. However, currently, the backing data cannot be stored in
    attributes called ``.values`` or ``._values`` to ensure full compatibility
    with pandas internals. But other names as ``.data``, ``._data``,
    ``._items``, ... can be freely used.

    If implementing NumPy's ``__array_ufunc__`` interface, pandas expects
    that

    1. You defer by returning ``NotImplemented`` when any Series are present
       in `inputs`. Pandas will extract the arrays and call the ufunc again.
    2. You define a ``_HANDLED_TYPES`` tuple as an attribute on the class.
       Pandas inspect this to determine whether the ufunc is valid for the
       types present.

    See :ref:`extending.extension.ufunc` for more.

    By default, ExtensionArrays are not hashable.  Immutable subclasses may
    override this behavior.

    Examples
    --------
    Please see the following:

    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/list/array.py
    """

    # '_typ' is for pandas.core.dtypes.generic.ABCExtensionArray.
    # Don't override this.
    _typ = "extension"

    # similar to __array_priority__, positions ExtensionArray after Index,
    #  Series, and DataFrame.  EA subclasses may override to choose which EA
    #  subclass takes priority. If overriding, the value should always be
    #  strictly less than 2000 to be below Index.__pandas_priority__.
    __pandas_priority__ = 1000

    _readonly = False

    # ------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: abstract class-level constructor. Parameters: scalars is a sequence of
        # scalars of the array's scalar type or convertible to it; dtype (keyword,
        # default None) a compatible dtype; copy (keyword, default false) whether to
        # copy data. Returns a new array. Subclasses must override; this raises an
        # abstract-method error.
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type in this method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray

        See Also
        --------
        api.extensions.ExtensionArray._from_sequence_of_strings : Construct a new
            ExtensionArray from a sequence of strings.
        api.extensions.ExtensionArray._hash_pandas_object : Hook for
            hash_pandas_object.

        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence([4, 5])
        <IntegerArray>
        [4, 5]
        Length: 2, dtype: Int64
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: abstract class-level constructor from strings, used by text parsers.
        # Parameters: strings is a sequence of strings; dtype (keyword, required) an
        # extension dtype; copy (keyword, default false). Optional for subclasses; this
        # default raises an abstract-method error.
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : ExtensionDtype
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray

        See Also
        --------
        api.extensions.ExtensionArray._from_sequence : Construct a new ExtensionArray
            from a sequence of scalars.
        api.extensions.ExtensionArray._from_factorized : Reconstruct an ExtensionArray
            after factorization.

        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence_of_strings(
        ...     ["1", "2", "3"], dtype=pd.Int64Dtype()
        ... )
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_factorized(cls, values, original):
        # [FUNCTION DOC]
        # HELPER: abstract class-level hook that rebuilds an array from factorization
        # output. Parameters: values is the integer-keyed array of unique values
        # returned by the values-for-factorize protocol; original is the array that was
        # factorized. Subclasses must override; this raises an abstract-method error.
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.
        ExtensionArray.factorize : Encode the extension array as an enumerated type.

        Examples
        --------
        >>> interv_arr = pd.arrays.IntervalArray(
        ...     [pd.Interval(0, 1), pd.Interval(1, 5), pd.Interval(1, 5)]
        ... )
        >>> codes, uniques = pd.factorize(interv_arr)
        >>> pd.arrays.IntervalArray._from_factorized(uniques, interv_arr)
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        # [FUNCTION DOC]
        # HELPER: class-level strict variant of the sequence constructor, used when
        # casting the result of an element-wise operation. Parameters: scalars is a
        # sequence of scalars; dtype (keyword) the target extension dtype. Calls the
        # sequence constructor without copying. Value and type errors propagate
        # unchanged; any other exception emits a warning (attributed to the caller's
        # stack level) saying the method should only raise those two, and is then re-
        # raised. Returns the new array.
        """
        Strict analogue to _from_sequence, allowing only sequences of scalars
        that should be specifically inferred to the given dtype.

        Parameters
        ----------
        scalars : sequence
        dtype : ExtensionDtype

        Raises
        ------
        TypeError or ValueError

        Notes
        -----
        This is called in a try/except block when casting the result of a
        pointwise operation in ExtensionArray._cast_pointwise_result.
        """
        try:
            return cls._from_sequence(scalars, dtype=dtype, copy=False)
        except (ValueError, TypeError):
            raise
        except Exception:
            warnings.warn(
                "_from_scalars should only raise ValueError or TypeError. "
                "Consider overriding _from_scalars where appropriate.",
                stacklevel=find_stack_level(),
            )
            raise

    def _cast_pointwise_result(self, values) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: converts the result sequence of an element-wise operation (such as a
        # map) into an array. Parameter values is the sequence. First tries the strict
        # class constructor using this array's own dtype; if that raises a value or type
        # error it falls back to an object ndarray run through generic type inference,
        # including non-numeric types. Returns an extension array or numpy array, not
        # necessarily of the same type as this array.
        """
        Construct an ExtensionArray after a pointwise operation.

        Cast the result of a pointwise operation (e.g. Series.map) to an
        array. This is not required to return an ExtensionArray of the same
        type as self or of the same dtype. It can also return another
        ExtensionArray of the same "family" if you implement multiple
        ExtensionArrays/Dtypes that are interoperable (e.g. if you have float
        array with units, this method can return an int array with units).

        If converting to your own ExtensionArray is not possible, this method
        falls back to returning an array with the default type inference.
        If you only need to cast to `self.dtype`, it is recommended to override
        `_from_scalars` instead of this method.

        Parameters
        ----------
        values : sequence

        Returns
        -------
        ExtensionArray or ndarray
        """
        try:
            return type(self)._from_scalars(values, dtype=self.dtype)
        except (ValueError, TypeError):
            values = np.asarray(values, dtype=object)
            return lib.maybe_convert_objects(values, convert_non_numeric=True)

    # ------------------------------------------------------------------------
    # Must be a Sequence
    # ------------------------------------------------------------------------
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    def __getitem__(self, item: PositionalIndexer) -> Self | Any:
        # [FUNCTION DOC]
        # Abstract item access. Parameter item is an integer, slice, boolean array or
        # integer list/array. Subclasses must return a scalar for an integer and a new
        # extension array (even for length zero or one) for all other indexers; this
        # default raises an abstract-method error.
        """
        Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.

            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None

            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

            * list[int]:  A list of int

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.

        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.

        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        raise AbstractMethodError(self)

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Optional in-place assignment, not required by the interface. Parameters: key
        # is an integer, integer array, boolean array or slice; value is a scalar or
        # sequence to assign. Raises a value error if the array is read-only; otherwise
        # the default raises a not-implemented error naming the type. The comments in
        # the body are guidance for implementers (support multiple keys, broadcast
        # scalars, coerce values like the constructor does).
        """
        Set one or more values inplace.

        This method is not required to satisfy the pandas extension array
        interface.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the array is readonly and modification is attempted.
        """
        # Some notes to the ExtensionArray implementer who may have ended up
        # here. While this method is not required for the interface, if you
        # *do* choose to implement __setitem__, then some semantics should be
        # observed:
        #
        # * Setting multiple values : ExtensionArrays should support setting
        #   multiple values at once, 'key' will be a sequence of integers and
        #  'value' will be a same-length sequence.
        #
        # * Broadcasting : For a sequence 'key' and a scalar 'value',
        #   each position in 'key' should be set to 'value'.
        #
        # * Coercion : Most users will expect basic coercion to work. For
        #   example, a string like '2018-01-01' is coerced to a datetime
        #   when setting on a datetime64ns array. In general, if the
        #   __init__ method coerces that value, then so should __setitem__
        # Note, also, that Series/DataFrame.where internally use __setitem__
        # on a copy of the data.
        # Check if the array is readonly
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        raise NotImplementedError(f"{type(self)} does not implement __setitem__.")

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Abstract length. Subclasses must return the number of elements as an integer;
        # this default raises an abstract-method error.
        """
        Length of this array

        Returns
        -------
        length : int
        """
        raise AbstractMethodError(self)

    def __iter__(self) -> Iterator[Any]:
        # [FUNCTION DOC]
        # Generator over the elements. The default yields each item by repeated indexing
        # from position zero to the length, which may be slower than necessary. Needed
        # so that the array is recognized as list-like.
        """
        Iterate over elements of the array.
        """
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: object) -> bool | np.bool_:
        # [FUNCTION DOC]
        # Membership test. Parameter item is any object. For a missing-value scalar it
        # avoids the comparison (which would give missing): returns false if the dtype
        # cannot hold missing values; otherwise true only if the item is the dtype's
        # missing value or an instance of the dtype's scalar type and the array has any
        # missing entries. For any other item it returns whether any element equals the
        # item using element-wise comparison.
        """
        Return for `item in self`.
        """
        # GH37867
        # comparisons of any item to pd.NA always return pd.NA, so e.g. "a" in [pd.NA]
        # would raise a TypeError. The implementation below works around that.
        if is_scalar(item) and isna(item):
            if not self._can_hold_na:
                return False
            elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                return self._hasna
            else:
                return False
        else:
            # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
            # attribute "any"
            return (item == self).any()  # type: ignore[union-attr]

    # error: Signature of "__eq__" incompatible with supertype "object"
    def __eq__(self, other: object) -> ArrayLike:  # type: ignore[override]
        # [FUNCTION DOC]
        # Abstract element-wise equality. Subclasses must return a boolean numpy array
        # or boolean extension array, and should return the not-implemented sentinel
        # when other is a series, index or dataframe. This default raises an abstract-
        # method error.
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        raise AbstractMethodError(self)

    # error: Signature of "__ne__" incompatible with supertype "object"
    def __ne__(self, other: object) -> ArrayLike:  # type: ignore[override]
        # [FUNCTION DOC]
        # Element-wise inequality, defined as the logical negation of the equality
        # result.
        """
        Return for `self != other` (element-wise in-equality).
        """
        # error: Unsupported operand type for ~ ("ExtensionArray")
        return ~(self == other)  # type: ignore[operator]

    def item(self, index: int | None = None):
        # [FUNCTION DOC]
        # Return one element as a scalar. Parameter index is an optional integer
        # position (default None). With no index the array must contain exactly one
        # element, which is returned, otherwise a value error is raised. With an index,
        # a non-integer raises a type error; otherwise the element at that position is
        # returned (an out-of-bounds position raises the index error from item access).
        # KNOWN ISSUE: the docstring does not document the type error raised for a non-
        # integer index.
        """
        Return the array element at the specified position as a Python scalar.

        Parameters
        ----------
        index : int, optional
            Position of the element. If not provided, the array must contain
            exactly one element.

        Returns
        -------
        scalar
            The element at the specified position.

        Raises
        ------
        ValueError
            If no index is provided and the array does not have exactly
            one element.
        IndexError
            If the specified position is out of bounds.

        See Also
        --------
        numpy.ndarray.item : Return the item of an array as a scalar.

        Examples
        --------
        >>> arr = pd.array([1], dtype="Int64")
        >>> arr.item()
        np.int64(1)

        >>> arr = pd.array([1, 2, 3], dtype="Int64")
        >>> arr.item(0)
        np.int64(1)
        >>> arr.item(2)
        np.int64(3)
        """
        if index is None:
            if len(self) != 1:
                raise ValueError(
                    "can only convert an array of size 1 to a Python scalar"
                )
            return self[0]
        else:
            if not is_integer(index):
                raise TypeError(f"index must be an integer, got {type(index)}")
            return self[index]

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Convert to a numpy array. Parameters: dtype (default None) target numpy dtype;
        # copy (default false) forces an independent copy; na_value (default the library
        # "no default" marker) a value to write into the positions that are missing.
        # Builds the result with the generic array conversion; copies if copy is true or
        # na_value is given; otherwise, if the array is read-only and converting to that
        # dtype would be a view, returns a read-only view. If na_value was given,
        # assigns it into the missing positions. Returns the numpy array.
        """
        Convert to a NumPy ndarray.

        This is similar to :meth:`numpy.asarray`, but may provide additional control
        over how the conversion is done.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

        Returns
        -------
        numpy.ndarray
        """
        result = np.asarray(self, dtype=dtype)
        if copy or na_value is not lib.no_default:
            result = result.copy()
        elif self._readonly and astype_is_view(self.dtype, result.dtype):
            # If the ExtensionArray is readonly, make the numpy array readonly too
            result = result.view()
            result.flags.writeable = False

        if na_value is not lib.no_default:
            result[self.isna()] = na_value  # type: ignore[index]

        return result

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @property
    def dtype(self) -> ExtensionDtype:
        # [FUNCTION DOC]
        # Abstract property: the extension dtype instance describing the array.
        # Subclasses must override; this raises an abstract-method error.
        """
        An instance of ExtensionDtype.

        See Also
        --------
        api.extensions.ExtensionDtype : Base class for extension dtypes.
        api.extensions.ExtensionArray : Base class for extension array types.
        api.extensions.ExtensionArray.dtype : The dtype of an ExtensionArray.
        Series.dtype : The dtype of a Series.
        DataFrame.dtype : The dtype of a DataFrame.

        Examples
        --------
        >>> pd.array([1, 2, 3]).dtype
        Int64Dtype()
        """
        raise AbstractMethodError(self)

    @property
    def shape(self) -> Shape:
        # [FUNCTION DOC]
        # Property: shape tuple of the array, here a one-element tuple holding the
        # length.
        """
        Return a tuple of the array dimensions.

        See Also
        --------
        numpy.ndarray.shape : Similar attribute which returns the shape of an array.
        DataFrame.shape : Return a tuple representing the dimensionality of the
            DataFrame.
        Series.shape : Return a tuple representing the dimensionality of the Series.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shape
        (3,)
        """
        return (len(self),)

    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property: number of elements, computed as the product of the shape entries.
        """
        The number of elements in the array.
        """
        # error: Incompatible return value type (got "signedinteger[_64Bit]",
        # expected "int")  [return-value]
        return np.prod(self.shape)  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC]
        # Property: number of dimensions; always one for the base class.
        """
        Extension Arrays are only allowed to be 1-dimensional.

        See Also
        --------
        ExtensionArray.shape: Return a tuple of the array dimensions.
        ExtensionArray.size: The number of elements in the array.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.ndim
        1
        """
        return 1

    @property
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Abstract property: bytes needed to store the data in memory (an approximate
        # lower bound is acceptable if exact size is expensive). Subclasses must
        # override; this raises an abstract-method error.
        """
        The number of bytes needed to store this object in memory.

        See Also
        --------
        ExtensionArray.shape: Return a tuple of the array dimensions.
        ExtensionArray.size: The number of elements in the array.

        Examples
        --------
        >>> pd.array([1, 2, 3]).nbytes
        27
        """
        # If this is expensive to compute, return an approximate lower bound
        # on the number of bytes needed.
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------------
    # Additional Methods
    # ------------------------------------------------------------------------

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        # [FUNCTION DOC]
        # Cast to another dtype. Parameters: dtype is a dtype-like (normalized first);
        # copy (default true) whether to always copy; when false a copy is made only if
        # needed. If the dtype equals the current dtype, returns self or a copy
        # according to copy. If the target is an extension dtype, builds an array of
        # that dtype's array class from this array. For numpy datetime64 or timedelta64
        # dtypes builds a datetime or timedelta array. Otherwise returns a numpy array
        # of that dtype (using the non-copying conversion when copy is false). Returns
        # an extension array or numpy array.
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        np.ndarray or pandas.api.extensions.ExtensionArray
            An ``ExtensionArray`` if ``dtype`` is ``ExtensionDtype``,
            otherwise a Numpy ndarray with ``dtype`` for its dtype.

        See Also
        --------
        Series.astype : Cast a Series to a different dtype.
        DataFrame.astype : Cast a DataFrame to a different dtype.
        api.extensions.ExtensionArray : Base class for ExtensionArray objects.
        core.arrays.DatetimeArray._from_sequence : Create a DatetimeArray from a
            sequence.
        core.arrays.TimedeltaArray._from_sequence : Create a TimedeltaArray from
            a sequence.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64

        Casting to another ``ExtensionDtype`` returns an ``ExtensionArray``:

        >>> arr1 = arr.astype("Float64")
        >>> arr1
        <FloatingArray>
        [1.0, 2.0, 3.0]
        Length: 3, dtype: Float64
        >>> arr1.dtype
        Float64Dtype()

        Otherwise, we will get a Numpy ndarray:

        >>> arr2 = arr.astype("float64")
        >>> arr2
        array([1., 2., 3.])
        >>> arr2.dtype
        dtype('float64')
        """
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if not copy:
                return self
            else:
                return self.copy()

        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            return cls._from_sequence(self, dtype=dtype, copy=copy)

        elif lib.is_np_dtype(dtype, "M"):
            from pandas.core.arrays import DatetimeArray

            return DatetimeArray._from_sequence(self, dtype=dtype, copy=copy)

        elif lib.is_np_dtype(dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(self, dtype=dtype, copy=copy)

        if not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)

    def isna(self) -> np.ndarray | ExtensionArraySupportsAnyAll:
        # [FUNCTION DOC]
        # Abstract missing-value detection. Subclasses must return a one-dimensional
        # boolean numpy array (or, for expensive cases such as sparse arrays, a boolean
        # extension array supporting any, all, reduce and accumulate) marking missing
        # positions. This default raises an abstract-method error.
        """
        A 1-D array indicating if each value is missing.

        Returns
        -------
        numpy.ndarray or pandas.api.extensions.ExtensionArray
            In most cases, this should return a NumPy ndarray. For
            exceptional cases like ``SparseArray``, where returning
            an ndarray would be expensive, an ExtensionArray may be
            returned.

        See Also
        --------
        ExtensionArray.dropna: Return ExtensionArray without NA values.
        ExtensionArray.fillna: Fill NA/NaN values using the specified method.

        Notes
        -----
        If returning an ExtensionArray, then

        * ``na_values._is_boolean`` should be True
        * ``na_values`` should implement :func:`ExtensionArray._reduce`
        * ``na_values`` should implement :func:`ExtensionArray._accumulate`
        * ``na_values.any`` and ``na_values.all`` should be implemented

        Examples
        --------
        >>> arr = pd.array([1, 2, np.nan, np.nan])
        >>> arr.isna()
        array([False, False,  True,  True])
        """
        raise AbstractMethodError(self)

    @property
    def _hasna(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property that says whether any element is missing, computed from the
        # missing mask. Subclasses may optimize it.
        # GH#22680
        """
        Equivalent to `self.isna().any()`.

        Some ExtensionArray subclasses may be able to optimize this check.
        """
        return bool(self.isna().any())

    def _values_for_argsort(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: returns the numpy array whose ordering is used for sorting, argmin and
        # argmax. Default is a numpy array made from this array. Entries at missing
        # positions need not sort correctly because callers handle the mask separately.
        # Callers must not modify the result.
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.

        Notes
        -----
        The caller is responsible for *not* modifying these values in-place, so
        it is safe for implementers to give views on ``self``.

        Functions that use this (e.g. ``ExtensionArray.argsort``) should ignore
        entries with missing values in the original array (according to
        ``self.isna()``). This means that the corresponding entries in the returned
        array don't need to be modified to sort correctly.

        Examples
        --------
        In most cases, this is the underlying Numpy array of the ``ExtensionArray``:

        >>> arr = pd.array([1, 2, 3])
        >>> arr._values_for_argsort()
        array([1, 2, 3])
        """
        # Note: this is used in `ExtensionArray.argsort/argmin/argmax`.
        return np.array(self)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        # [FUNCTION DOC]
        # Indices that would sort the array. Keyword-only parameters: ascending (default
        # true); kind (default quicksort) sort algorithm; na_position (first or last,
        # default last) where missing values go; extra keyword arguments are validated
        # against numpy's argsort signature (the ascending value is validated through
        # that check). Sorts the values from the values-for-argsort hook with the shared
        # sorting helper, using the missing mask. Returns an integer index array.
        """
        Return the indices that would sort this array.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        na_position : {'first', 'last'}, default 'last'
            If ``'first'``, put ``NaN`` values at the beginning.
            If ``'last'``, put ``NaN`` values at the end.
        **kwargs
            Passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Array of indices that sort ``self``. If NaN values are contained,
            NaN values are placed at the end.

        See Also
        --------
        numpy.argsort : Sorting implementation used internally.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argsort()
        array([1, 2, 0, 4, 3])
        """
        # Implementer note: You have two places to override the behavior of
        # argsort.
        # 1. _values_for_argsort : construct the values passed to np.argsort
        # 2. argsort : total control over sorting. In case of overriding this,
        #    it is recommended to also override argmax/argmin
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)

        values = self._values_for_argsort()
        return nargsort(
            values,
            kind=kind,
            ascending=ascending,
            na_position=na_position,
            mask=np.asarray(self.isna()),
        )

    def argmin(self, skipna: bool = True) -> int:
        # [FUNCTION DOC]
        # Position of the minimum (first occurrence on ties). Parameter skipna (default
        # true, validated as boolean). Raises a value error if skipna is false and any
        # value is missing. Uses the shared sorting helper on this array.
        """
        Return the index of minimum value.

        In case of multiple occurrences of the minimum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        int

        See Also
        --------
        ExtensionArray.argmax : Return the index of the maximum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmin()
        np.int64(1)
        """
        # Implementer note: You have two places to override the behavior of
        # argmin.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmin itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return nargminmax(self, "argmin")

    def argmax(self, skipna: bool = True) -> int:
        # [FUNCTION DOC]
        # Position of the maximum (first occurrence on ties). Parameter skipna (default
        # true, validated as boolean). Raises a value error if skipna is false and any
        # value is missing. Uses the shared sorting helper on this array.
        """
        Return the index of maximum value.

        In case of multiple occurrences of the maximum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        int

        See Also
        --------
        ExtensionArray.argmin : Return the index of the minimum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmax()
        np.int64(3)
        """
        # Implementer note: You have two places to override the behavior of
        # argmax.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmax itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return nargminmax(self, "argmax")

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        # [FUNCTION DOC]
        # Fill missing values by interpolation (optional; base class not supported). All
        # parameters are keyword-only and are the same as for the series interpolate
        # method: method, axis, index, limit, limit_direction, limit_area, copy, plus
        # extra keywords for the interpolating function. Subclasses should return an
        # array of their own type even when copy is false. This default always raises a
        # not-implemented error naming the type.
        """
        Fill NaN values using an interpolation method.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. One of:
            * 'linear': Ignore the index and treat the values as equally spaced.
            This is the only method supported on MultiIndexes.
            * 'time': Works on daily and higher resolution data to interpolate
            given length of interval.
            * 'index', 'values': use the actual numerical values of the index.
            * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric',
            'polynomial': Passed to scipy.interpolate.interp1d, whereas 'spline'
            is passed to scipy.interpolate.UnivariateSpline. These methods use
            the numerical values of the index.
            Both 'polynomial' and 'spline' require that you also specify an
            order (int), e.g. arr.interpolate(method='polynomial', order=5).
            * 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
            'cubicspline': Wrappers around the SciPy interpolation methods
            of similar names. See Notes.
            * 'from_derivatives': Refers to scipy.interpolate.BPoly.from_derivatives.
        axis : int
            Axis to interpolate along. For 1-dimensional data, use 0.
        index : Index
            Index to use for interpolation.
        limit : int or None
            Maximum number of consecutive NaNs to fill. Must be greater than 0.
        limit_direction : {'forward', 'backward', 'both'}
            Consecutive NaNs will be filled in this direction.
        limit_area : {'inside', 'outside'} or None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.
            * None: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).
        copy : bool
            If True, a copy of the object is returned with interpolated values.
        **kwargs : optional
            Keyword arguments to pass on to the interpolating function.

        Returns
        -------
        ExtensionArray
            An ExtensionArray with interpolated values.

        See Also
        --------
        Series.interpolate : Interpolate values in a Series.
        DataFrame.interpolate : Interpolate values in a DataFrame.

        Notes
        -----
        - All parameters must be specified as keyword arguments.
        - The 'krogh', 'piecewise_polynomial', 'spline', 'pchip' and 'akima'
          methods are wrappers around the respective SciPy implementations of
          similar names. These use the actual numerical values of the index.

        Examples
        --------
        Interpolating values in a NumPy array:

        >>> arr = pd.arrays.NumpyExtensionArray(np.array([0, 1, np.nan, 3]))
        >>> arr.interpolate(
        ...     method="linear",
        ...     limit=3,
        ...     limit_direction="forward",
        ...     index=pd.Index(range(len(arr))),
        ...     fill_value=1,
        ...     copy=False,
        ...     axis=0,
        ...     limit_area="inside",
        ... )
        <NumpyExtensionArray>
        [0.0, 1.0, 2.0, 3.0]
        Length: 4, dtype: float64

        Interpolating values in a FloatingArray:

        >>> arr = pd.array([1.0, pd.NA, 3.0, 4.0, pd.NA, 6.0], dtype="Float64")
        >>> arr.interpolate(
        ...     method="linear",
        ...     axis=0,
        ...     index=pd.Index(range(len(arr))),
        ...     limit=None,
        ...     limit_direction="both",
        ...     limit_area=None,
        ...     copy=True,
        ... )
        <FloatingArray>
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        Length: 6, dtype: Float64
        """
        # NB: we return type(self) even if copy=False
        raise NotImplementedError(
            f"{type(self).__name__} does not implement interpolate"
        )

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # HELPER: forward or backward fill used by the series and dataframe fill
        # methods. Keyword-only parameters: method (pad/ffill or backfill/bfill), limit
        # (default None, the maximum consecutive fills), limit_area (inside, outside or
        # None), copy (default true). If nothing is missing, returns self when copy is
        # false else a copy. Otherwise the method name is normalized and the fill
        # indexer is computed from the missing mask (restricted by limit_area when
        # given); forward fill takes by that indexer with filling allowed, backward fill
        # does the same on the reversed array and so also reverses the indexer. The base
        # implementation ignores the copy keyword when values are missing and always
        # returns a new array.
        """
        Pad or backfill values, used by Series/DataFrame ffill and bfill.

        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

        limit : int, default None
            This is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

        limit_area : {'inside', 'outside'} or None, default None
            Specifies which area to limit filling.
            - 'inside': Limit the filling to the area within the gaps.
            - 'outside': Limit the filling to the area outside the gaps.
            If `None`, no limitation is applied.

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword if any NAs are
            present.

        Returns
        -------
        Same type as self
            The filled array with the same type as the original.

        See Also
        --------
        Series.ffill : Forward fill missing values.
        Series.bfill : Backward fill missing values.
        DataFrame.ffill : Forward fill missing values in DataFrame.
        DataFrame.bfill : Backward fill missing values in DataFrame.
        api.types.isna : Check for missing values.
        api.types.isnull : Check for missing values.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr._pad_or_backfill(method="backfill", limit=1)
        <IntegerArray>
        [<NA>, 2, 2, 3, <NA>, <NA>]
        Length: 6, dtype: Int64
        """
        mask = self.isna()

        if mask.any():
            # NB: the base class does not respect the "copy" keyword
            meth = missing.clean_fill_method(method)

            npmask = np.asarray(mask)
            if limit_area is not None and not npmask.all():
                _fill_limit_area_1d(npmask, limit_area)
            if meth == "pad":
                indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                return self.take(indexer, allow_fill=True)
            else:
                # i.e. meth == "backfill"
                indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                return self[::-1].take(indexer, allow_fill=True)

        else:
            if not copy:
                return self
            new_values = self.copy()
        return new_values

    def fillna(
        self,
        value: object | ArrayLike,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Fill missing values. Parameters: value is a scalar or an array-like of the
        # same length; limit (default None) the maximum number of missing entries to
        # fill, counted from the start; copy (default true). The mask is trimmed to the
        # first limit missing entries when the limit is smaller than the length, and the
        # value's size is checked against it. Then a copy (or a slice sharing data when
        # copy is false) has the masked entries assigned the value (using item
        # assignment, which the subclass must support). Returns the filled array of the
        # same type.
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It's expected
            that the array-like have the same length as 'self'.
        limit : int, default None
            The maximum number of entries where NA values will be filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        See Also
        --------
        api.extensions.ExtensionArray.dropna : Return ExtensionArray without
            NA values.
        api.extensions.ExtensionArray.isna : A 1-D array indicating if
            each value is missing.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        """
        mask = self.isna()
        if limit is not None and limit < len(self):
            # isna can return an ExtensionArray, we're assuming that comparisons
            # are implemented.
            # mypy doesn't like that mask can be an EA which need not have `cumsum`
            modify = mask.cumsum() > limit  # type: ignore[union-attr]
            if modify.any():
                # Only copy mask if necessary
                mask = mask.copy()
                mask[modify] = False
        # error: Argument 2 to "check_value_size" has incompatible type
        # "ExtensionArray"; expected "ndarray"
        value = missing.check_value_size(
            value,
            mask,  # type: ignore[arg-type]
            len(self),
        )

        if mask.any():
            # fill with value
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
            new_values[mask] = value
        elif not copy:
            new_values = self[:]
        else:
            new_values = self.copy()
        return new_values

    def dropna(self) -> Self:
        # [FUNCTION DOC]
        # Return a new array without the missing entries, by indexing with the inverted
        # missing mask.
        """
        Return ExtensionArray without NA values.

        Returns
        -------
        Self
            An ExtensionArray of the same type as the original but with all
            NA values removed.

        See Also
        --------
        Series.dropna : Remove missing values from a Series.
        DataFrame.dropna : Remove missing values from a DataFrame.
        api.extensions.ExtensionArray.isna : Check for missing values in
            an ExtensionArray.

        Examples
        --------
        >>> pd.array([1, 2, np.nan]).dropna()
        <IntegerArray>
        [1, 2]
        Length: 2, dtype: Int64
        """
        # error: Unsupported operand type for ~ ("ExtensionArray")
        return self[~self.isna()]  # type: ignore[operator]

    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Boolean numpy array marking duplicated values. Parameter keep: first (default,
        # first occurrence not marked), last, or false (all duplicates marked). Uses the
        # shared duplicated routine with the missing mask.
        """
        Return boolean ndarray denoting duplicate values.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        ndarray[bool]
            With true in indices where elements are duplicated and false otherwise.

        See Also
        --------
        DataFrame.duplicated : Return boolean Series denoting
            duplicate rows.
        Series.duplicated : Indicate duplicate Series values.
        api.extensions.ExtensionArray.unique : Compute the ExtensionArray
            of unique values.

        Examples
        --------
        >>> pd.array([1, 1, 2, 3, 3], dtype="Int64").duplicated()
        array([False,  True, False, False,  True])
        """
        mask = self.isna().astype(np.bool_, copy=False)
        return duplicated(values=self, keep=keep, mask=mask)

    def shift(self, periods: int = 1, fill_value: object = None) -> ExtensionArray:
        # [FUNCTION DOC]
        # Shift values by a number of positions, filling vacated positions. Parameters:
        # periods (default 1; negative shifts backward) and fill_value (default None,
        # meaning the dtype's missing value; any missing value is replaced by it). An
        # empty array or zero periods returns a copy. Otherwise builds a filler array of
        # the fill value with length equal to the smaller of the shift size and array
        # length, and concatenates filler and the retained slice in the right order.
        # Assumes the dtype's missing value can be stored in the array. Returns an array
        # of the same type.
        """
        Shift values by desired number.

        Newly introduced missing values are filled with
        ``self.dtype.na_value``.

        Parameters
        ----------
        periods : int, default 1
            The number of periods to shift. Negative values are allowed
            for shifting backwards.

        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            The default is ``self.dtype.na_value``.

        Returns
        -------
        ExtensionArray
            Shifted.

        See Also
        --------
        api.extensions.ExtensionArray.transpose : Return a transposed view on
            this array.
        api.extensions.ExtensionArray.factorize : Encode the extension array as an
            enumerated type.

        Notes
        -----
        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is
        returned.

        If ``periods > len(self)``, then an array of size
        len(self) is returned, with all values filled with
        ``self.dtype.na_value``.

        For 2-dimensional ExtensionArrays, we are always shifting along axis=0.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shift(2)
        <IntegerArray>
        [<NA>, <NA>, 1]
        Length: 3, dtype: Int64
        """
        # Note: this implementation assumes that `self.dtype.na_value` can be
        # stored in an instance of your ExtensionArray with `self.dtype`.
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=self.dtype
        )
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods) :]
            b = empty
        return self._concat_same_type([a, b])

    def unique(self) -> Self:
        # [FUNCTION DOC]
        # Unique values in order of appearance. Default implementation converts to
        # object dtype, takes uniques with the generic routine and rebuilds an array of
        # this dtype.
        """
        Compute the ExtensionArray of unique values.

        Returns
        -------
        pandas.api.extensions.ExtensionArray
            With unique values from the input array.

        See Also
        --------
        Index.unique: Return unique values in the index.
        Series.unique: Return unique values of Series object.
        unique: Return unique values based on a hash table.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 1, 2, 3])
        >>> arr.unique()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        uniques = unique(self.astype(object))
        return self._from_sequence(uniques, dtype=self.dtype)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # [FUNCTION DOC]
        # Insertion positions that would keep the array sorted. Parameters: value
        # (scalar, array-like or extension array); side (left, default, or right);
        # sorter (optional integer index array that sorts the data). Default converts
        # self (and an extension array value) to object dtype and uses the numpy search
        # on that. Returns an integer position or array of positions.
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
        # Note: the base tests provided by pandas only test the basics.
        # We do not test
        # 1. Values outside the range of the `data_for_sorting` fixture
        # 2. Values between the values in the `data_for_sorting` fixture
        # 3. Missing values.
        arr = self.astype(object)
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        return arr.searchsorted(value, side=side, sorter=sorter)

    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Whether another object is an equivalent array. Returns false if the types
        # differ, the dtypes differ or the lengths differ. Otherwise compares element-
        # wise (missing results of a nullable comparison count as unequal), and
        # considers positions where both arrays are missing equal; true only if every
        # position is equal. Returns a bool.
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        See Also
        --------
        numpy.array_equal : Equivalent method for numpy array.
        Series.equals : Equivalent method for Series.
        DataFrame.equals : Equivalent method for DataFrame.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True

        >>> arr1 = pd.array([1, 3, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        False
        """
        if type(self) != type(other):
            return False
        other = cast(ExtensionArray, other)
        if self.dtype != other.dtype:
            return False
        elif len(self) != len(other):
            return False
        else:
            equal_values = self == other
            if isinstance(equal_values, ExtensionArray):
                # boolean array with NA -> fill with False
                equal_values = equal_values.fillna(False)
            # error: Unsupported left operand type for & ("ExtensionArray")
            equal_na = self.isna() & other.isna()  # type: ignore[operator]
            return bool((equal_values | equal_na).all())

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Membership test for each element. Parameter values is an array (numpy or
        # extension) of values to look for. Default converts self to a numpy array and
        # uses the generic membership routine, returning a boolean numpy array.
        # Subclasses may return their own boolean array type.
        """
        Pointwise comparison for set containment in the given values.

        Roughly equivalent to `np.array([x in values for x in self])`

        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            Values to compare every element in the array against.

        Returns
        -------
        np.ndarray[bool]
            With true at indices where value is in `values`.

        See Also
        --------
        DataFrame.isin: Whether each element in the DataFrame is contained in values.
        Index.isin: Return a boolean array where the index values are in values.
        Series.isin: Whether elements in Series are contained in values.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.isin([1])
        <BooleanArray>
        [True, False, False]
        Length: 3, dtype: boolean
        """
        return isin(np.asarray(self), values)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        # [FUNCTION DOC]
        # HELPER: returns a pair (values, na_value) for factorization and hashing. The
        # values are this array cast to object dtype by default, and the missing marker
        # is NaN by default. Entries equal to the marker are treated as missing and
        # coded as the negative sentinel code.
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `-1` and not included in `uniques`. By default,
            ``np.nan`` is used.

        See Also
        --------
        util.hash_pandas_object : Hash the pandas object.

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`. If needed, this can be
        overridden in the ``self._hash_pandas_object()`` method.

        Examples
        --------
        >>> pd.array([1, 2, 3])._values_for_factorize()
        (array([1, 2, 3], dtype=object), nan)
        """
        return self.astype(object), np.nan

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        # [FUNCTION DOC]
        # Encode the array as integer codes plus unique values. Parameter
        # use_na_sentinel (default true): when true missing values get the code -1 and
        # are excluded from the uniques; when false they get a regular non-negative code
        # and appear in the uniques. Obtains values and the missing marker from the
        # values-for-factorize hook, runs the generic array factorization, and rebuilds
        # the uniques as an extension array through the factorized reconstruction hook.
        # Returns a tuple of (integer codes numpy array, unique values array).
        """
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

        Returns
        -------
        codes : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.

        Examples
        --------
        >>> idx1 = pd.PeriodIndex(
        ...     ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
        ...     freq="M",
        ... )
        >>> arr, idx = idx1.factorize()
        >>> arr
        array([0, 0, 1, 1, 2, 2])
        >>> idx
        PeriodIndex(['2014-01', '2014-02', '2014-03'], dtype='period[M]')
        """
        # Implementer note: There are two ways to override the behavior of
        # pandas.factorize
        # 1. _values_for_factorize and _from_factorize.
        #    Specify the values passed to pandas' internal factorization
        #    routines, and how to convert from those values back to the
        #    original ExtensionArray.
        # 2. ExtensionArray.factorize.
        #    Complete control over factorization.
        arr, na_value = self._values_for_factorize()

        codes, uniques = factorize_array(
            arr, use_na_sentinel=use_na_sentinel, na_value=na_value
        )

        uniques_ea = self._from_factorized(uniques, self)
        return codes, uniques_ea

    _extension_array_shared_docs["repeat"] = """
        Repeat elements of a %(klass)s.

        Returns a new %(klass)s where each element of the current %(klass)s
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            %(klass)s.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        %(klass)s
            Newly created %(klass)s with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.
        ExtensionArray.take : Take arbitrary positions.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, str): ['a', 'b', 'c']
        >>> cat.repeat(2)
        ['a', 'a', 'b', 'b', 'c', 'c']
        Categories (3, str): ['a', 'b', 'c']
        >>> cat.repeat([1, 2, 3])
        ['a', 'b', 'b', 'c', 'c', 'c']
        Categories (3, str): ['a', 'b', 'c']
        """

    def repeat(self, repeats: int | Sequence[int], axis: AxisInt | None = None) -> Self:
        # [FUNCTION DOC]
        # Repeat each element consecutively. Parameters: repeats is a non-negative
        # integer or sequence of integers (one per element); axis must be None and is
        # only accepted for numpy compatibility (validated, so other values raise).
        # Implemented by repeating positions and taking them. Returns a new array of the
        # same type.
        """
        Repeat elements of an ExtensionArray.

        Returns a new ExtensionArray where each element of the current ExtensionArray
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            ExtensionArray.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        ExtensionArray
            Newly created ExtensionArray with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.
        ExtensionArray.take : Take arbitrary positions.

        Examples
        --------
        >>> cat = pd.Categorical(["a", "b", "c"])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, str): ['a', 'b', 'c']
        >>> cat.repeat(2)
        ['a', 'a', 'b', 'b', 'c', 'c']
        Categories (3, str): ['a', 'b', 'c']
        >>> cat.repeat([1, 2, 3])
        ['a', 'b', 'b', 'c', 'c', 'c']
        Categories (3, str): ['a', 'b', 'c']
        """
        nv.validate_repeat((), {"axis": axis})
        ind = np.arange(len(self)).repeat(repeats)
        return self.take(ind)

    def value_counts(self, dropna: bool = True) -> Series:
        # [FUNCTION DOC]
        # Counts of unique values as a series. Parameter dropna (default true) excludes
        # missing values. Converts to numpy without copying, counts without sorting
        # using the generic routine, then casts the resulting index to this array's
        # dtype. Returns a series indexed by the unique values.
        """
        Return a Series containing counts of unique values.

        This method returns a Series with unique values as the index and their
        counts as the values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NA values.

        Returns
        -------
        Series
            A Series with unique values as index and counts as values.

        See Also
        --------
        Series.value_counts : Equivalent method on Series.
        DataFrame.value_counts : Equivalent method on DataFrame.
        Index.value_counts : Equivalent method on Index.

        Examples
        --------
        >>> from pandas.core.arrays import IntegerArray
        >>> arr = IntegerArray._from_sequence([3, 3, 3, 1, 2, 2])
        >>> arr.value_counts()
        3    3
        1    1
        2    2
        Name: count, dtype: Int64
        """
        from pandas.core.algorithms import value_counts_internal as value_counts

        result = value_counts(self.to_numpy(copy=False), sort=False, dropna=dropna)
        result.index = result.index.astype(self.dtype)
        return result

    # ------------------------------------------------------------------------
    # Indexing methods
    # ------------------------------------------------------------------------

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Abstract positional selection. Parameters: indices (sequence or one-
        # dimensional integer array); allow_fill (keyword, default false): when false
        # negative indices count from the end, when true -1 marks positions to fill and
        # other negatives raise a value error; fill_value (keyword, default None meaning
        # the dtype's missing value) is the user-facing scalar used for filled positions
        # and must be translated to physical storage by the implementation. Out-of-
        # bounds indices raise an index error. Returns a new array of the same type.
        # Subclasses must override; this raises an abstract-method error.
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int or one-dimensional np.ndarray of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray
            An array formed with selected `indices`.

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take : Take elements from an array along an axis.
        api.extensions.take : Take elements from an array.

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.

        Examples
        --------
        Here's an example implementation, which relies on casting the
        extension array to object dtype. This uses the helper method
        :func:`pandas.api.extensions.take`.

        .. code-block:: python

           def take(self, indices, allow_fill=False, fill_value=None):
               from pandas.api.extensions import take

               # If the ExtensionArray is backed by an ndarray, then
               # just pass that here instead of coercing to object.
               data = self.astype(object)

               if allow_fill and fill_value is None:
                   fill_value = self.dtype.na_value

               # fill value should always be translated from the scalar
               # type for the array, to the physical storage type for
               # the data, before passing to take.

               result = take(
                   data, indices, fill_value=fill_value, allow_fill=allow_fill
               )
               return self._from_sequence(result, dtype=self.dtype)
        """
        # Implementer note: The `fill_value` parameter should be a user-facing
        # value, an instance of self.dtype.type. When passed `fill_value=None`,
        # the default of `self.dtype.na_value` should be used.
        # This may differ from the physical storage type your ExtensionArray
        # uses. In this case, your implementation is responsible for casting
        # the user-facing type to the storage type, before using
        # pandas.api.extensions.take
        raise AbstractMethodError(self)

    def copy(self) -> Self:
        # [FUNCTION DOC]
        # Abstract copy. Subclasses must return a new array of the same type whose
        # changes do not affect the original; this raises an abstract-method error.
        """
        Return a copy of the array.

        This method creates a copy of the `ExtensionArray` where modifying the
        data in the copy will not affect the original array. This is useful when
        you want to manipulate data without altering the original dataset.

        Returns
        -------
        ExtensionArray
            A new `ExtensionArray` object that is a copy of the current instance.

        See Also
        --------
        DataFrame.copy : Return a copy of the DataFrame.
        Series.copy : Return a copy of the Series.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.copy()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        raise AbstractMethodError(self)

    @overload
    def view(self) -> Self:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def view(self, dtype: Dtype | None = ...) -> ArrayLike:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # [FUNCTION DOC]
        # Return a view on the data. Parameter dtype (default None). Subclasses must
        # return a new object (not self) referencing the same data. The base
        # implementation supports only no dtype, returning a full slice of itself; any
        # dtype raises a not-implemented error carrying the requested dtype.
        """
        Return a view on the array.

        Parameters
        ----------
        dtype : str, np.dtype, or ExtensionDtype, optional
            Default None.

        Returns
        -------
        ExtensionArray or np.ndarray
            A view on the :class:`ExtensionArray`'s data.

        See Also
        --------
        api.extensions.ExtensionArray.ravel: Return a flattened view on input array.
        Index.view: Equivalent function for Index.
        ndarray.view: New view of array with the same data.

        Examples
        --------
        This gives view on the underlying data of an ``ExtensionArray`` and is not a
        copy. Modifications on either the view or the original ``ExtensionArray``
        will be reflected on the underlying data:

        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.view()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [2, 2, 3]
        Length: 3, dtype: Int64
        """
        # NB:
        # - This must return a *new* object referencing the same data, not self.
        # - The only case that *must* be implemented is with dtype=None,
        #   giving a view with the same dtype as self.
        if dtype is not None:
            raise NotImplementedError(dtype)
        return self[:]

    # ------------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # String representation. For arrays with more than one dimension delegates to
        # the 2D representation. Otherwise formats the elements with the shared object-
        # summary formatter (using this array's scalar formatter, trailing separators
        # stripped), and returns a text with a first line holding the class name in
        # angle brackets, the element summary, and a footer line with length and dtype.
        if self.ndim > 1:
            return self._repr_2d()

        from pandas.io.formats.printing import format_object_summary

        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False
        ).rstrip(", \n")
        class_name = f"<{type(self).__name__}>\n"
        footer = self._get_repr_footer()
        return f"{class_name}{data}\n{footer}"

    def _get_repr_footer(self) -> str:
        # [FUNCTION DOC]
        # HELPER: footer line for the representation: shape and dtype for multi-
        # dimensional arrays, length and dtype otherwise.
        # GH#24278
        if self.ndim > 1:
            return f"Shape: {self.shape}, dtype: {self.dtype}"
        return f"Length: {len(self)}, dtype: {self.dtype}"

    def _repr_2d(self) -> str:
        # [FUNCTION DOC]
        # HELPER: representation for arrays with more than one dimension. Formats each
        # row with the shared summary formatter, joins the rows with commas and newlines
        # inside square brackets under the class name line, and adds the footer line.
        from pandas.io.formats.printing import format_object_summary

        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        lines = [
            format_object_summary(x, self._formatter(), indent_for_name=False).rstrip(
                ", \n"
            )
            for x in self
        ]
        data = ",\n".join(lines)
        class_name = f"<{type(self).__name__}>"
        footer = self._get_repr_footer()
        return f"{class_name}\n[\n{data}\n]\n{footer}"

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # [FUNCTION DOC]
        # Returns the function used to render individual scalars in the representation.
        # Parameter boxed (default false) says whether the array is shown inside a
        # series, dataframe or index. The default returns the built-in text conversion
        # when boxed and the built-in representation conversion otherwise. Subclasses
        # override to customize.
        """
        Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed : bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.

        See Also
        --------
        api.extensions.ExtensionArray._concat_same_type : Concatenate multiple
            array of this dtype.
        api.extensions.ExtensionArray._explode : Transform each element of
            list-like to a row.
        api.extensions.ExtensionArray._from_factorized : Reconstruct an
            ExtensionArray after factorization.
        api.extensions.ExtensionArray._from_sequence : Construct a new
            ExtensionArray from a sequence of scalars.

        Examples
        --------
        >>> class MyExtensionArray(pd.arrays.NumpyExtensionArray):
        ...     def _formatter(self, boxed=False):
        ...         return lambda x: "*" + str(x) + "*"
        >>> MyExtensionArray(np.array([1, 2, 3, 4]))
        <MyExtensionArray>
        [*1*, *2*, *3*, *4*]
        Length: 4, dtype: int64
        """
        if boxed:
            return str
        return repr

    # ------------------------------------------------------------------------
    # Reshaping
    # ------------------------------------------------------------------------

    def transpose(self, *axes: int) -> Self:
        # [FUNCTION DOC]
        # Transposed view. Because arrays are one-dimensional this is a no-op returning
        # a full slice. The positional axes arguments are accepted for numpy
        # compatibility and ignored.
        """
        Return a transposed view on this array.

        Because ExtensionArrays are always 1D, this is a no-op.  It is included
        for compatibility with np.ndarray.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> pd.array([1, 2, 3]).transpose()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self[:]

    @property
    def T(self) -> Self:
        # [FUNCTION DOC]
        # Property: the transpose of the array (see the transpose method).
        return self.transpose()

    def ravel(self, order: Literal["C", "F", "A", "K"] | None = "C") -> Self:
        # [FUNCTION DOC]
        # Flattened view. Parameter order is accepted for numpy compatibility and
        # ignored (default C). Because arrays are one-dimensional it returns self.
        """
        Return a flattened view on this array.

        Parameters
        ----------
        order : {None, 'C', 'F', 'A', 'K'}, default 'C'

        Returns
        -------
        ExtensionArray
            A flattened view on the array.

        See Also
        --------
        ExtensionArray.tolist: Return a list of the values.

        Notes
        -----
        - Because ExtensionArrays are 1D-only, this is a no-op.
        - The "order" argument is ignored, is for compatibility with NumPy.

        Examples
        --------
        >>> pd.array([1, 2, 3]).ravel()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        # [FUNCTION DOC]
        # HELPER: abstract class-level concatenation. Parameter to_concat is a sequence
        # of arrays of this class, all with the same dtype (guaranteed by callers). Must
        # return a new array of the same dtype. Subclasses must override; this raises an
        # abstract-method error.
        """
        Concatenate multiple array of this dtype.

        Parameters
        ----------
        to_concat : sequence of this type
            An array of the same dtype to concatenate.

        Returns
        -------
        ExtensionArray

        See Also
        --------
        api.extensions.ExtensionArray._explode : Transform each element of
            list-like to a row.
        api.extensions.ExtensionArray._formatter : Formatting function for
            scalar values.
        api.extensions.ExtensionArray._from_factorized : Reconstruct an
            ExtensionArray after factorization.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, 3])
        >>> arr2 = pd.array([4, 5, 6])
        >>> pd.arrays.IntegerArray._concat_same_type([arr1, arr2])
        <IntegerArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: Int64
        """
        # Implementer note: this method will only be called with a sequence of
        # ExtensionArrays of this class and with the same dtype as self. This
        # should allow "easy" concatenation (no upcasting needed), and result
        # in a new ExtensionArray of the same dtype.
        # Note: this strict behaviour is only guaranteed starting with pandas 1.1
        raise AbstractMethodError(cls)

    # The _can_hold_na attribute is set to True so that pandas internals
    # will use the ExtensionDtype.na_value as the NA value in operations
    # such as take(), reindex(), shift(), etc.  In addition, those results
    # will then be of the ExtensionArray subclass rather than an array
    # of objects
    @cache_readonly
    def _can_hold_na(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached property saying whether the array can hold missing values,
        # taken from the dtype. When true, pandas internals use the dtype's missing
        # value for operations such as take, reindex and shift and keep results as this
        # array type.
        return self.dtype._can_hold_na

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ExtensionArray:
        # [FUNCTION DOC]
        # HELPER: optional hook for cumulative operations. Parameters: name is one of
        # the cumulative min, max, sum or product names; skipna (keyword, default true);
        # extra keywords passed to the operation. The result dtype may differ. This
        # default raises a not-implemented error naming the operation and dtype.
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array
            An array performing the accumulation operation.

        Raises
        ------
        NotImplementedError : subclass does not define accumulations

        See Also
        --------
        api.extensions.ExtensionArray._concat_same_type : Concatenate multiple
            array of this dtype.
        api.extensions.ExtensionArray.view : Return a view on the array.
        api.extensions.ExtensionArray._explode : Transform each element of
            list-like to a row.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr._accumulate(name="cumsum")
        <IntegerArray>
        [1, 3, 6]
        Length: 3, dtype: Int64
        """
        raise NotImplementedError(f"cannot perform {name} with type {self.dtype}")

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: reduction dispatcher. Parameters: name is the reduction name (any,
        # all, min, max, sum, mean, median, prod, std, var, sem, kurt, skew); skipna
        # (keyword, default true); keepdims (keyword, default false); extra keywords
        # passed to the method. Looks up a method of that name on the array and raises a
        # type error if there is none. Calls it with skipna and the extra keywords. With
        # keepdims, the min and max results are wrapped in a new array of this type and
        # all other results in a one-element numpy array.
        # KNOWN ISSUE: the docstring says that with keepdims the result is a numpy array
        # with a single element, but for min and max the code returns an extension array
        # of this type.
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        keepdims : bool, default False
            If False, a scalar is returned.
            If True, the result has dimension with size one along the reduced axis.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar or ndarray:
            The result of the reduction operation. The type of the result
            depends on `keepdims`:
            - If `keepdims` is `False`, a scalar value is returned.
            - If `keepdims` is `True`, the result is wrapped in a numpy array with
            a single element.

        Raises
        ------
        TypeError : subclass does not define operations

        See Also
        --------
        Series.min : Return the minimum value.
        Series.max : Return the maximum value.
        Series.sum : Return the sum of values.
        Series.mean : Return the mean of values.
        Series.median : Return the median of values.
        Series.std : Return the standard deviation.
        Series.var : Return the variance.
        Series.prod : Return the product of values.
        Series.sem : Return the standard error of the mean.
        Series.kurt : Return the kurtosis.
        Series.skew : Return the skewness.

        Examples
        --------
        >>> pd.array([1, 2, 3])._reduce("min")
        np.int64(1)
        >>> pd.array([1, 2, 3])._reduce("max")
        np.int64(3)
        >>> pd.array([1, 2, 3])._reduce("sum")
        np.int64(6)
        >>> pd.array([1, 2, 3])._reduce("mean")
        np.float64(2.0)
        >>> pd.array([1, 2, 3])._reduce("median")
        np.float64(2.0)
        """
        meth = getattr(self, name, None)
        if meth is None:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support operation '{name}'"
            )
        result = meth(skipna=skipna, **kwargs)
        if keepdims:
            if name in ["min", "max"]:
                result = self._from_sequence([result], dtype=self.dtype)
            else:
                result = np.array([result])

        return result

    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    __hash__: ClassVar[None]  # type: ignore[assignment]

    # ------------------------------------------------------------------------
    # Non-Optimized Default Methods; in the case of the private methods here,
    #  these are not guaranteed to be stable across pandas versions.

    def _values_for_json(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: values used when serializing to JSON. Default is a numpy array of the
        # data. Subclasses with types that the JSON encoder cannot handle may return an
        # array of strings.
        """
        Specify how to render our entries in to_json.

        Notes
        -----
        The dtype on the returned ndarray is not restricted, but for non-native
        types that are not specifically handled in objToJSON.c, to_json is
        liable to raise. In these cases, it may be safer to return an ndarray
        of strings.
        """
        return np.asarray(self)

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        # [FUNCTION DOC]
        # HELPER: hook for hashing pandas objects. Keyword-only parameters: encoding
        # (string encoding for text data), hash_key (string key for hashing), categorize
        # (whether to categorize object arrays first for speed). The default hashes the
        # values returned by the values-for-factorize hook with the generic array hasher
        # (imported lazily). Returns an unsigned 64-bit integer array.
        """
        Hook for hash_pandas_object.

        Default is to use the values returned by _values_for_factorize.

        Parameters
        ----------
        encoding : str
            Encoding for data & key when strings.
        hash_key : str
            Hash_key for string key to encode.
        categorize : bool
            Whether to first categorize object arrays before hashing. This is more
            efficient when the array contains duplicate values.

        Returns
        -------
        np.ndarray[uint64]
            An array of hashed values.

        See Also
        --------
        api.extensions.ExtensionArray._values_for_factorize : Return an array and
            missing value suitable for factorization.
        util.hash_array : Given a 1d array, return an array of hashed values.

        Examples
        --------
        >>> pd.array([1, 2])._hash_pandas_object(
        ...     encoding="utf-8", hash_key="1000000000000000", categorize=False
        ... )
        array([ 6238072747940578789, 15839785061582574730], dtype=uint64)
        """
        from pandas.core.util.hashing import hash_array

        values, _ = self._values_for_factorize()
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    def _explode(self) -> tuple[Self, npt.NDArray[np.uint64]]:
        # [FUNCTION DOC]
        # HELPER: support for exploding list-like elements into rows. Default returns a
        # copy of the array and an unsigned 64-bit array of ones with the array's length
        # (meaning each element is unchanged). Subclasses with list-like elements
        # override; the second return value gives the number of output rows each
        # original element produced.
        """
        Transform each element of list-like to a row.

        For arrays that do not contain list-like elements the default
        implementation of this method just returns a copy and an array
        of ones (unchanged index).

        Returns
        -------
        ExtensionArray
            Array with the exploded values.
        np.ndarray[uint64]
            The original lengths of each list-like for determining the
            resulting index.

        See Also
        --------
        Series.explode : The method on the ``Series`` object that this
            extension array method is meant to support.

        Examples
        --------
        >>> import pyarrow as pa
        >>> a = pd.array(
        ...     [[1, 2, 3], [4], [5, 6]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))
        ... )
        >>> a._explode()
        (<ArrowExtensionArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: int64[pyarrow], array([3, 1, 2], dtype=int32))
        """
        values = self.copy()
        counts = np.ones(shape=(len(self),), dtype=np.uint64)
        return values, counts

    def tolist(self) -> list:
        # [FUNCTION DOC]
        # Return the values as a Python list. For multi-dimensional arrays, returns a
        # nested list built from each row; otherwise a list built by iterating the
        # array.
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list
            Python list of values in array.

        See Also
        --------
        Index.to_list: Return a list of the values in the Index.
        Series.to_list: Return a list of the values in the Series.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.tolist()
        [1, 2, 3]
        """
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self)

    def delete(self, loc: PositionalIndexer) -> Self:
        # [FUNCTION DOC]
        # Return a new array without the elements at the given position(s). Parameter
        # loc is an integer, slice or sequence of integers. Computes the remaining
        # positions with the numpy delete and takes them. Used by index deletion.
        indexer = np.delete(np.arange(len(self)), loc)
        return self.take(indexer)

    def insert(self, loc: int, item) -> Self:
        # [FUNCTION DOC]
        # Return a new array with an item inserted. Parameters: loc is the integer
        # position (validated and normalized against the length); item is a scalar. The
        # item is turned into a one-element array with the class constructor (which must
        # raise for invalid items, normally a value or type error), then the head slice,
        # the item and the tail slice are concatenated with the class concatenation.
        # Result keeps type and dtype.
        """
        Insert an item at the given position.

        Parameters
        ----------
        loc : int
            Index where the `item` needs to be inserted.
        item : scalar-like
            Value to be inserted.

        Returns
        -------
        ExtensionArray
            With `item` inserted at `loc`.

        See Also
        --------
        Index.insert: Make new Index inserting new item at location.

        Notes
        -----
        This method should be both type and dtype-preserving.  If the item
        cannot be held in an array of this type/dtype, either ValueError or
        TypeError should be raised.

        The default implementation relies on _from_sequence to raise on invalid
        items.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.insert(2, -1)
        <IntegerArray>
        [1, 2, -1, 3]
        Length: 4, dtype: Int64
        """
        loc = validate_insert_loc(loc, len(self))

        item_arr = type(self)._from_sequence([item], dtype=self.dtype)

        return type(self)._concat_same_type([self[:loc], item_arr, self[loc:]])

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # [FUNCTION DOC]
        # HELPER: in-place analogue of the numpy masked assignment. Parameters: mask is
        # a boolean array; value is a scalar or a list-like of the same length as the
        # array (not repeated cyclically as numpy would). For list-likes only the
        # entries at the mask positions are used. Assigns through item assignment.
        # Returns nothing.
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike
            If listlike, must be arraylike with same length as self.

        Returns
        -------
        None

        Notes
        -----
        Unlike np.putmask, we do not repeat listlike values with mismatched length.
        'value' should either be a scalar or an arraylike with the same length
        as self.
        """
        if is_list_like(value):
            val = value[mask]
        else:
            val = value

        self[mask] = val

    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        # [FUNCTION DOC]
        # HELPER: analogue of the numpy conditional select. Parameters: mask is a
        # boolean array (true keeps the current value); value is a scalar or list-like
        # of the same length used where the mask is false. Works on a copy by assigning
        # to the inverted mask positions. Returns an array of the same type.
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Returns
        -------
        same type as self
        """
        result = self.copy()

        if is_list_like(value):
            val = value[~mask]
        else:
            val = value

        result[~mask] = val
        return result

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        # [FUNCTION DOC]
        # HELPER: ranking support; behaves like the series rank method. Keyword-only
        # parameters: axis (only zero supported, otherwise raises a not-implemented
        # error); method (default average), na_option (default keep), ascending (default
        # true) and pct (default false) as for the series method. Ranks the values-for-
        # argsort data with the generic rank routine, passing the missing mask only if
        # there are missing values. Returns the rank array.
        """
        See Series.rank.__doc__.
        """
        if axis != 0:
            raise NotImplementedError

        return rank(
            self._values_for_argsort(),
            axis=axis,
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
            mask=np.asarray(self.isna(), dtype="bool") if self._hasna else None,
        )

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype):
        # [FUNCTION DOC]
        # HELPER: class-level analogue of the numpy uninitialized-array constructor.
        # Parameters: shape (tuple of ints) and dtype (extension dtype). Makes an empty
        # array of the dtype and takes a broadcast array of -1 indices with filling
        # allowed, so all entries are missing. Raises a not-implemented error if the
        # result is not an instance of the class or has a different dtype than
        # requested. Called by the dtype's public empty method, which is the official
        # API; subclasses must still define this.
        """
        Create an ExtensionArray with the given shape and dtype.

        See also
        --------
        ExtensionDtype.empty
            ExtensionDtype.empty is the 'official' public version of this API.
        """
        # Implementer note: while ExtensionDtype.empty is the public way to
        # call this method, it is still required to implement this `_empty`
        # method as well (it is called internally in pandas)
        obj = cls._from_sequence([], dtype=dtype)

        taker = np.broadcast_to(np.intp(-1), shape)
        result = obj.take(taker, allow_fill=True)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )
        return result

    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        # [FUNCTION DOC]
        # HELPER: quantile computation. Parameters: qs (float array of quantiles) and
        # interpolation (method name). Converts the data to numpy, passes the missing
        # mask and NaN as the fill value to the masked quantile routine, and builds a
        # new array of this type from the result using the class sequence constructor
        # (without a dtype). Returns an array of the same type.
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
        interpolation: str

        Returns
        -------
        same type as self
        """
        mask = np.asarray(self.isna())
        arr = np.asarray(self)
        fill_value = np.nan

        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        return type(self)._from_sequence(res_values)

    def _mode(self, dropna: bool = True) -> Self:
        # [FUNCTION DOC]
        # HELPER: modes of the array. Parameter dropna (default true) excludes missing
        # values from counting. Uses the generic mode routine and returns only the
        # values part, always as an array (even if there is one mode), sorted when
        # possible.
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NA values.

        Returns
        -------
        same type as self
            Sorted, if possible.
        """
        # error: Incompatible return value type (got "Union[ExtensionArray,
        # Tuple[np.ndarray, npt.NDArray[np.bool_]]", expected "Self")
        result, _ = mode(self, dropna=dropna)
        return result  # type: ignore[return-value]

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # [FUNCTION DOC]
        # Numpy ufunc protocol. Parameters: ufunc, the method name, and the inputs and
        # keyword arguments. Returns the not-implemented sentinel if any input is a
        # series, index or dataframe (so pandas unwraps them and retries). Otherwise
        # tries, in order: mapping the ufunc to the corresponding dunder operator; the
        # out-parameter dispatch if an output was supplied; for reduce, the reduction
        # dispatch; and finally the default ufunc implementation, which applies the
        # ufunc to numpy conversions of the inputs.
        if any(
            isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)) for other in inputs
        ):
            return NotImplemented

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        return arraylike.default_array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        # [FUNCTION DOC]
        # Map values through a function, dict or series. Parameters: mapper; na_action
        # (None, default, or ignore, which propagates missing values without calling the
        # mapper; subclasses not supporting ignore should raise a not-implemented
        # error). Delegates to the generic map routine. Returns a numpy array, index or
        # extension array (a multi-index if the function returns multi-element tuples).
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence. If 'ignore' is not supported, a
            ``NotImplementedError`` should be raised.

        Returns
        -------
        Union[ndarray, Index, ExtensionArray]
            The output of the mapping function applied to the array.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        return map_array(self, mapper, na_action=na_action)

    # ------------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: experimental hook for group-wise reductions and transformations.
        # Keyword-only parameters: how is the operation name; has_dropped_na says
        # whether missing group keys were dropped; min_count is the minimum number of
        # valid values; ngroups is the group count; ids is the integer group label of
        # each element; extra keywords are operation specific (skipna, ddof, rank
        # options). Only string dtypes are supported by the base class; any other dtype
        # raises a not-implemented error. For strings, product, mean, median, cumulative
        # sum and product, standard deviation, standard error, variance, skew and
        # kurtosis raise a type error; other operations other than any and all are
        # checked up front that a compiled implementation exists for object data. The
        # data is converted to an object numpy array with NaN for missing. For sum the
        # initial value is the empty string and, if skipna is requested with min_count
        # zero, missing values are first filled with the empty string (a skipna keyword
        # is required, asserted). The compiled group operation is then run. Rank results
        # are returned as is; other results are rebuilt as an array of the string dtype.
        """
        Dispatch GroupBy reduction or transformation operation.

        This is an *experimental* API to allow ExtensionArray authors to implement
        reductions and transformations. The API is subject to change.

        Parameters
        ----------
        how : {'any', 'all', 'sum', 'prod', 'min', 'max', 'mean', 'median',
               'median', 'var', 'std', 'sem', 'nth', 'last', 'ohlc',
               'cumprod', 'cumsum', 'cummin', 'cummax', 'rank'}
        has_dropped_na : bool
        min_count : int
        ngroups : int
        ids : np.ndarray[np.intp]
            ids[i] gives the integer label for the group that self[i] belongs to.
        **kwargs : operation-specific
            'any', 'all' -> ['skipna']
            'var', 'std', 'sem' -> ['ddof']
            'cumprod', 'cumsum', 'cummin', 'cummax' -> ['skipna']
            'rank' -> ['ties_method', 'ascending', 'na_option', 'pct']

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        from pandas.core.arrays.string_ import StringDtype
        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        initial: Any = 0
        # GH#43682
        if isinstance(self.dtype, StringDtype):
            # StringArray
            if op.how in [
                "prod",
                "mean",
                "median",
                "cumsum",
                "cumprod",
                "std",
                "sem",
                "var",
                "skew",
                "kurt",
            ]:
                raise TypeError(
                    f"dtype '{self.dtype}' does not support operation '{how}'"
                )
            if op.how not in ["any", "all"]:
                # Fail early to avoid conversion to object
                op._get_cython_function(op.kind, op.how, np.dtype(object), False)

            arr = self
            if op.how == "sum":
                initial = ""
                # https://github.com/pandas-dev/pandas/issues/60229
                # All NA should result in the empty string.
                assert "skipna" in kwargs
                if kwargs["skipna"] and min_count == 0:
                    arr = arr.fillna("")
            npvalues = arr.to_numpy(object, na_value=np.nan)
        else:
            raise NotImplementedError(
                f"function is not implemented for this dtype: {self.dtype}"
            )

        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=None,
            initial=initial,
            **kwargs,
        )

        if op.how in op.cast_blocklist:
            # i.e. how in ["rank"], since other cast_blocklist methods don't go
            #  through cython_operation
            return res_values

        if isinstance(self.dtype, StringDtype):
            dtype = self.dtype
            string_array_cls = dtype.construct_array_type()
            return string_array_cls._from_sequence(res_values, dtype=dtype)

        else:
            raise NotImplementedError


class ExtensionArraySupportsAnyAll(ExtensionArray):
    # [CLASS DOC]
    # Typing-oriented subclass of the extension array base declaring that the array
    # supports any and all reductions (used for extension arrays returned by the
    # missing-value mask method). Both methods are abstract here and have overloads that
    # type the return as plain bool when missing values are skipped and as bool-or-NA
    # otherwise.
    @overload
    def any(self, *, skipna: Literal[True] = ...) -> bool:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def any(self, *, skipna: bool) -> bool | NAType:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    def any(self, *, skipna: bool = True) -> bool | NAType:
        # [FUNCTION DOC]
        # Abstract any-reduction. Keyword parameter skipna (default true). Subclasses
        # return whether any element is true. Raises an abstract-method error here.
        raise AbstractMethodError(self)

    @overload
    def all(self, *, skipna: Literal[True] = ...) -> bool:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    @overload
    def all(self, *, skipna: bool) -> bool | NAType:
        # [FUNCTION DOC]
        # Typing-only overload stub: declares one accepted call signature for static
        # type checking and has no behavior. See the implementation that follows the
        # overloads for the real behavior.
        ...

    def all(self, *, skipna: bool = True) -> bool | NAType:
        # [FUNCTION DOC]
        # Abstract all-reduction. Keyword parameter skipna (default true). Subclasses
        # return whether all elements are true. Raises an abstract-method error here.
        raise AbstractMethodError(self)


class ExtensionOpsMixin:
    # [CLASS DOC]
    # Base mixin that links operator implementations to their special-method names on a
    # class. Subclasses supply factory methods that create the arithmetic, comparison
    # and logical operator methods; the class methods here attach (by attribute
    # assignment on the class) the generated methods for each operator and its reflected
    # form. A class using it may want to set a numpy array priority so numpy defers to
    # it.
    """
    A base class for linking the operators to their dunder names.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_arithmetic_method(cls, op):
        # [FUNCTION DOC]
        # HELPER: abstract factory. Parameter op is an operator function. Subclasses
        # return a method implementing it. Raises an abstract-method error here.
        raise AbstractMethodError(cls)

    @classmethod
    def _add_arithmetic_ops(cls) -> None:
        # [FUNCTION DOC]
        # HELPER: class-level installer. Takes no arguments, returns nothing. Sets on
        # the class the add, subtract, multiply, power, modulo, floor-divide, true-
        # divide and divmod special methods and their reflected counterparts, each
        # created through the arithmetic factory. Mutates the class.
        setattr(cls, "__add__", cls._create_arithmetic_method(operator.add))
        setattr(cls, "__radd__", cls._create_arithmetic_method(roperator.radd))
        setattr(cls, "__sub__", cls._create_arithmetic_method(operator.sub))
        setattr(cls, "__rsub__", cls._create_arithmetic_method(roperator.rsub))
        setattr(cls, "__mul__", cls._create_arithmetic_method(operator.mul))
        setattr(cls, "__rmul__", cls._create_arithmetic_method(roperator.rmul))
        setattr(cls, "__pow__", cls._create_arithmetic_method(operator.pow))
        setattr(cls, "__rpow__", cls._create_arithmetic_method(roperator.rpow))
        setattr(cls, "__mod__", cls._create_arithmetic_method(operator.mod))
        setattr(cls, "__rmod__", cls._create_arithmetic_method(roperator.rmod))
        setattr(cls, "__floordiv__", cls._create_arithmetic_method(operator.floordiv))
        setattr(
            cls, "__rfloordiv__", cls._create_arithmetic_method(roperator.rfloordiv)
        )
        setattr(cls, "__truediv__", cls._create_arithmetic_method(operator.truediv))
        setattr(cls, "__rtruediv__", cls._create_arithmetic_method(roperator.rtruediv))
        setattr(cls, "__divmod__", cls._create_arithmetic_method(divmod))
        setattr(cls, "__rdivmod__", cls._create_arithmetic_method(roperator.rdivmod))

    @classmethod
    def _create_comparison_method(cls, op):
        # [FUNCTION DOC]
        # HELPER: abstract factory. Parameter op is an operator function. Subclasses
        # return a method implementing the comparison. Raises an abstract-method error
        # here.
        raise AbstractMethodError(cls)

    @classmethod
    def _add_comparison_ops(cls) -> None:
        # [FUNCTION DOC]
        # HELPER: class-level installer. Sets the equal, not-equal, less, greater, less-
        # equal and greater-equal special methods on the class, each created through the
        # comparison factory. Mutates the class; returns nothing.
        setattr(cls, "__eq__", cls._create_comparison_method(operator.eq))
        setattr(cls, "__ne__", cls._create_comparison_method(operator.ne))
        setattr(cls, "__lt__", cls._create_comparison_method(operator.lt))
        setattr(cls, "__gt__", cls._create_comparison_method(operator.gt))
        setattr(cls, "__le__", cls._create_comparison_method(operator.le))
        setattr(cls, "__ge__", cls._create_comparison_method(operator.ge))

    @classmethod
    def _create_logical_method(cls, op):
        # [FUNCTION DOC]
        # HELPER: abstract factory. Parameter op is an operator function. Subclasses
        # return a method implementing the logical operation. Raises an abstract-method
        # error here.
        raise AbstractMethodError(cls)

    @classmethod
    def _add_logical_ops(cls) -> None:
        # [FUNCTION DOC]
        # HELPER: class-level installer. Sets the and, or, xor special methods and their
        # reflected counterparts on the class, each created through the logical factory.
        # Mutates the class; returns nothing.
        setattr(cls, "__and__", cls._create_logical_method(operator.and_))
        setattr(cls, "__rand__", cls._create_logical_method(roperator.rand_))
        setattr(cls, "__or__", cls._create_logical_method(operator.or_))
        setattr(cls, "__ror__", cls._create_logical_method(roperator.ror_))
        setattr(cls, "__xor__", cls._create_logical_method(operator.xor))
        setattr(cls, "__rxor__", cls._create_logical_method(roperator.rxor))


@set_module("pandas.api.extensions")
class ExtensionScalarOpsMixin(ExtensionOpsMixin):
    # [CLASS DOC]
    # Public mixin (extensions API namespace) that implements operators for an extension
    # array by applying the operator to the underlying scalar elements one by one. It
    # extends the operator-linking base mixin; arithmetic operators try to convert the
    # result back to the array type, comparisons return plain boolean numpy arrays. A
    # subclass of an extension array lists this mixin as an additional base and calls
    # the installer class methods after the class definition.
    """
    A mixin for defining ops on an ExtensionArray.

    It is assumed that the underlying scalar objects have the operators
    already defined.

    Notes
    -----
    If you have defined a subclass MyExtensionArray(ExtensionArray), then
    use MyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin) to
    get the arithmetic operators.  After the definition of MyExtensionArray,
    insert the lines

    MyExtensionArray._add_arithmetic_ops()
    MyExtensionArray._add_comparison_ops()

    to link the operators to your class.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_method(cls, op, coerce_to_dtype: bool = True, result_dtype=None):
        # [FUNCTION DOC]
        # HELPER: class-level factory that builds an operator method. Parameters: op is
        # a binary operator function; coerce_to_dtype (default true) says whether to try
        # to convert the result back to the array's type (falling back to numpy);
        # result_dtype (default None) is the dtype used for the numpy result when not
        # coercing. Returns a method, named after the operator via the name-setting
        # helper, that takes the other operand and applies the operator element by
        # element.
        # KNOWN ISSUE: the docstring does not document the result_dtype parameter.
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.

        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype : bool, default True
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype.
            If it's not possible to create a new ExtensionArray with the
            values, an ndarray is returned instead.

        Returns
        -------
        Callable[[Any, Any], Union[ndarray, ExtensionArray]]
            A method that can be bound to a class. When used, the method
            receives the two arguments, one of which is the instance of
            this class, and should return an ExtensionArray or an ndarray.

            Returning an ndarray may be necessary when the result of the
            `op` cannot be stored in the ExtensionArray. The dtype of the
            ndarray uses NumPy's normal inference rules.

        Examples
        --------
        Given an ExtensionArray subclass called MyExtensionArray, use

            __add__ = cls._create_method(operator.add)

        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """

        def _binop(self, other):
            # [FUNCTION DOC]
            # HELPER: the generated operator method. Parameter other is the right
            # operand. Returns the not-implemented sentinel for series, index and
            # dataframe operands. Otherwise broadcasts a non-list-like scalar to the
            # array length, applies the operator to the pairs of elements (strict
            # pairing, so a length mismatch raises a value error; the operator's own
            # type error propagates), and converts the list of results. For divmod and
            # its reflected form, the paired results are split into two sequences and
            # converted separately, returning a pair.
            def convert_values(param):
                # [FUNCTION DOC]
                # HELPER: nested function that returns the operand as is if it is an
                # extension array or list-like, otherwise a list repeating the scalar
                # once per element of the array.
                if isinstance(param, ExtensionArray) or is_list_like(param):
                    ovalues = param
                else:  # Assume its an object
                    ovalues = [param] * len(self)
                return ovalues

            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            lvalues = self
            rvalues = convert_values(other)

            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = [op(a, b) for (a, b) in zip(lvalues, rvalues, strict=True)]

            def _maybe_convert(arr):
                # [FUNCTION DOC]
                # HELPER: nested function converting a list of element results. When
                # coercion is enabled, it casts through the pointwise-result hook and
                # falls back to a numpy array if the result is not the same type as the
                # array. When disabled, it makes a numpy array with the configured
                # result dtype. Returns the converted array.
                if coerce_to_dtype:
                    # https://github.com/pandas-dev/pandas/issues/22850
                    # We catch all regular exceptions here, and fall back
                    # to an ndarray.
                    res = self._cast_pointwise_result(arr)
                    if not isinstance(res, type(self)):
                        # exception raised in _from_sequence; ensure we have ndarray
                        res = np.asarray(arr)
                else:
                    res = np.asarray(arr, dtype=result_dtype)
                return res

            if op.__name__ in {"divmod", "rdivmod"}:
                a, b = zip(*res, strict=True)
                return _maybe_convert(a), _maybe_convert(b)

            return _maybe_convert(res)

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        # [FUNCTION DOC]
        # HELPER: implements the arithmetic factory by creating a coercing operator
        # method. Parameter op is the operator function.
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        # [FUNCTION DOC]
        # HELPER: implements the comparison factory by creating a non-coercing operator
        # method whose numpy result has boolean dtype. Parameter op is the operator
        # function.
        return cls._create_method(op, coerce_to_dtype=False, result_dtype=bool)
