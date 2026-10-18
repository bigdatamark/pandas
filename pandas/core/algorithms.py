"""
Generic data algorithms. This module is experimental at the moment and not
intended for public consumption
"""

# [MODULE DOC]
# Core generic algorithms operating on array-likes: hashtable-based uniqueness,
# factorization, membership, value counts, duplicate detection, mode, ranking,
# take, difference, searchsorted, safe sorting, multiset union and mapping through
# a lookup. Works for numpy arrays, extension arrays, Index and Series inputs by
# unwrapping them first and re-wrapping results.
# Public contents: unique (also aliased as unique1d), unique_with_mask,
# nunique_ints, isin, factorize, factorize_array, value_counts_internal,
# value_counts_arraylike, duplicated, mode, rank, take, searchsorted, diff,
# safe_sort, union_with_duplicates and map_array (unique, factorize and take are
# exposed in the public API under the top-level or extension namespaces through the
# module-setting decorator). Private helpers (names starting with an underscore)
# handle coercion to hashtable-friendly dtypes, reconstruction, validation, hashtable
# selection and mixed-type sorting.
# Relationships: relies on the hashtable and algos extension modules, the dtype
# helpers, the take algorithms and construction helpers; used by Series, Index,
# DataFrame, groupby and extension arrays.
# Import-time side effects: builds the dtype-name to hashtable-class lookup table.
from __future__ import annotations

import decimal
import operator
from typing import (
    TYPE_CHECKING,
    Literal,
    TypeVar,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    algos,
    hashtable as htable,
    iNaT,
    lib,
)
from pandas._libs.missing import NA
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    ArrayLikeT,
    AxisInt,
    DtypeObj,
    TakeIndexer,
    npt,
)
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    np_find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_object,
    ensure_platform_int,
    is_bool_dtype,
    is_complex_dtype,
    is_dict_like,
    is_dtype_equal,
    is_extension_array_dtype,
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_signed_integer_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    CategoricalDtype,
    ExtensionDtype,
    NumpyEADtype,
)
from pandas.core.dtypes.generic import (
    ABCDatetimeArray,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCNumpyExtensionArray,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import validate_indices

if TYPE_CHECKING:
    from pandas._typing import (
        ListLike,
        NumpySorter,
        NumpyValueArrayLike,
    )

    from pandas import (
        Categorical,
        Index,
        Series,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        ExtensionArray,
    )

    T = TypeVar("T", bound=Index | Categorical | ExtensionArray)


# --------------- #
# dtype access    #
# --------------- #
def _ensure_data(values: ArrayLike) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: coerces array data to a numpy dtype that the low-level hashtable and
    # algorithm routines support. Input: values, numpy array, extension array or
    # (a MultiIndex, which is not unwrapped). Behavior by dtype: object arrays stay
    # object; masked arrays without missing entries use their raw data (recursively
    # coerced), otherwise they become an object array; categoricals become their
    # integer codes; booleans become unsigned 8-bit integers; integers and most
    # floats are used as is (float16 and the 12/16-byte floats are converted to
    # 64-bit floats); complex stays complex; datetime-like data is viewed as 64-bit
    # integers; anything else is converted to object dtype. Returns a numpy array.
    # No exceptions are raised by the function itself.
    """
    routine to ensure that our data is of the correct
    input dtype for lower-level routines

    This will coerce:
    - ints -> int64
    - uint -> uint64
    - bool -> uint8
    - datetimelike -> i8
    - datetime64tz -> i8 (in local tz)
    - categorical -> codes

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    np.ndarray
    """

    if not isinstance(values, ABCMultiIndex):
        # extract_array would raise
        values = extract_array(values, extract_numpy=True)

    if is_object_dtype(values.dtype):
        return ensure_object(np.asarray(values))

    elif isinstance(values.dtype, BaseMaskedDtype):
        # i.e. BooleanArray, FloatingArray, IntegerArray
        values = cast("BaseMaskedArray", values)
        if not values._hasna:
            # No pd.NAs -> We can avoid an object-dtype cast (and copy) GH#41816
            #  recurse to avoid re-implementing logic for eg bool->uint8
            return _ensure_data(values._data)
        return np.asarray(values)

    elif isinstance(values.dtype, CategoricalDtype):
        # NB: cases that go through here should NOT be using _reconstruct_data
        #  on the back-end.
        values = cast("Categorical", values)
        return values.codes

    elif is_bool_dtype(values.dtype):
        if isinstance(values, np.ndarray):
            # i.e. actually dtype == np.dtype("bool")
            return np.asarray(values).view("uint8")
        else:
            # e.g. Sparse[bool, False]  # TODO: no test cases get here
            return np.asarray(values).astype("uint8", copy=False)

    elif is_integer_dtype(values.dtype):
        return np.asarray(values)

    elif is_float_dtype(values.dtype):
        # Note: checking `values.dtype == "float128"` raises on Windows and 32bit
        # error: Item "ExtensionDtype" of "Union[Any, ExtensionDtype, dtype[Any]]"
        # has no attribute "itemsize"
        if values.dtype.itemsize in [2, 12, 16]:  # type: ignore[union-attr]
            # we dont (yet) have float128 hashtable support
            return ensure_float64(values)
        return np.asarray(values)

    elif is_complex_dtype(values.dtype):
        return cast(np.ndarray, values)

    # datetimelike
    elif needs_i8_conversion(values.dtype):
        npvalues = values.view("i8")
        npvalues = cast(np.ndarray, npvalues)
        return npvalues

    # we have failed, return object
    values = np.asarray(values, dtype=object)
    return ensure_object(values)


def _reconstruct_data(
    values: ArrayLikeT, dtype: DtypeObj, original: AnyArrayLike
) -> ArrayLikeT:
    # [FUNCTION DOC]
    # HELPER: undoes the coercion done to make data hashtable-friendly. Inputs:
    # values (result array in the coerced representation), dtype (the original
    # dtype) and original (the original array-like, unused here). If values is
    # already an extension array of the target dtype it is returned as is.
    # For a non-numpy target dtype the extension array class of that dtype is
    # built from the values; for a numpy target dtype the values are converted
    # without copying when possible. Returns the reconstructed array.
    """
    reverse of _ensure_data

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    original : AnyArrayLike

    Returns
    -------
    ExtensionArray or np.ndarray
    """
    if isinstance(values, ABCExtensionArray) and values.dtype == dtype:
        # Catch DatetimeArray/TimedeltaArray
        return values

    if not isinstance(dtype, np.dtype):
        # i.e. ExtensionDtype; note we have ruled out above the possibility
        #  that values.dtype == dtype
        cls = dtype.construct_array_type()

        # error: Incompatible return value type
        # (got "ExtensionArray",
        # expected "ndarray[tuple[Any, ...], dtype[Any]]")
        return cls._from_sequence(values, dtype=dtype)  # type: ignore[return-value]

    # error: Incompatible return value type
    # (got "ndarray[tuple[Any, ...], dtype[Any]]",
    # expected "ExtensionArray")
    return values.astype(dtype, copy=False)  # type: ignore[return-value]


def _ensure_arraylike(values, func_name: str) -> ArrayLike:
    # [FUNCTION DOC]
    # HELPER: validates and converts the input of an algorithm to an array-like.
    # Inputs: values and func_name (used in the error message and as a flag).
    # Index, Series, extension array and numpy array inputs are returned as is.
    # Any other input raises a type error naming the function and the input type,
    # except when func_name is the special isin-targets marker, in which case the
    # input is converted: mixed, string and mixed-integer inferred contents become
    # an object array (tuples are first turned into lists), anything else is
    # converted with the default array conversion. Returns the array-like.
    """
    ensure that we are arraylike if not already
    """
    if not isinstance(
        values,
        (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray, ABCNumpyExtensionArray),
    ):
        # GH#52986
        if func_name != "isin-targets":
            # Make an exception for the comps argument in isin.
            raise TypeError(
                f"{func_name} requires a Series, Index, "
                f"ExtensionArray, np.ndarray or NumpyExtensionArray "
                f"got {type(values).__name__}."
            )

        inferred = lib.infer_dtype(values, skipna=False)
        if inferred in ["mixed", "string", "mixed-integer"]:
            # "mixed-integer" to ensure we do not cast ["ss", 42] to str GH#22160
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values


_hashtables = {
    "complex128": htable.Complex128HashTable,
    "complex64": htable.Complex64HashTable,
    "float64": htable.Float64HashTable,
    "float32": htable.Float32HashTable,
    "uint64": htable.UInt64HashTable,
    "uint32": htable.UInt32HashTable,
    "uint16": htable.UInt16HashTable,
    "uint8": htable.UInt8HashTable,
    "int64": htable.Int64HashTable,
    "int32": htable.Int32HashTable,
    "int16": htable.Int16HashTable,
    "int8": htable.Int8HashTable,
    "string": htable.StringHashTable,
    "object": htable.PyObjectHashTable,
}


def _get_hashtable_algo(
    values: np.ndarray,
) -> tuple[type[htable.HashTable], np.ndarray]:
    # [FUNCTION DOC]
    # HELPER: picks the hashtable class for the data. Input: values (array). The
    # data is coerced with the data-coercion helper, the table key is the dtype name
    # (or the string table for object arrays holding only strings), and the matching
    # class is looked up in the module level table (a key error if the dtype has
    # no entry). Returns a pair: the hashtable class (not an instance) and the
    # coerced values.
    """
    Parameters
    ----------
    values : np.ndarray

    Returns
    -------
    htable : HashTable subclass
    values : ndarray
    """
    values = _ensure_data(values)

    ndtype = _check_object_for_strings(values)
    hashtable = _hashtables[ndtype]
    return hashtable, values


def _check_object_for_strings(values: np.ndarray) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the hashtable key name for an array: its dtype name, except
    # that an object array consisting entirely of strings (missing values counted
    # as non-strings) is reported as the string key, since the string hashtable is
    # cheaper than the object one.
    """
    Check if we can use string hashtable instead of object hashtable.

    Parameters
    ----------
    values : ndarray

    Returns
    -------
    str
    """
    ndtype = values.dtype.name
    if ndtype == "object":
        # it's cheaper to use a String Hash Table than Object; we infer
        # including nulls because that is the only difference between
        # StringHashTable and ObjectHashtable
        if lib.is_string_array(values, skipna=False):
            ndtype = "string"
    return ndtype


# --------------- #
# top-level algos #
# --------------- #


@overload
def unique(values: T) -> T:
    # [FUNCTION DOC]
    # Typing overload stub: Index, Categorical and extension array in, same type out.
    # See the implementation below.
    ...
@overload
def unique(values: np.ndarray | Series) -> np.ndarray:
    # [FUNCTION DOC]
    # Typing overload stub: numpy array or Series in, numpy array out. See the
    # implementation below.
    ...


@set_module("pandas")
def unique(values):
    # [FUNCTION DOC]
    # Public top-level function. Input: values, a one dimensional Series, Index,
    # extension array or numpy array (other input types such as plain lists raise
    # a type error). Returns the distinct values in order of first appearance,
    # including missing values, without sorting; for an Index an Index, for a
    # categorical or other extension data an extension array, otherwise a numpy
    # array. Simply forwards to the masked variant with no mask.
    """
    Return unique values based on a hash table.

    Uniques are returned in order of appearance. This does NOT sort.

    Significantly faster than numpy.unique for long enough sequences.
    Includes NA values.

    Parameters
    ----------
    values : 1d array-like
        The input array-like object containing values from which to extract
        unique values.

    Returns
    -------
    numpy.ndarray, ExtensionArray or NumpyExtensionArray

        The return can be:

        * Index : when the input is an Index
        * Categorical : when the input is a Categorical dtype
        * ndarray : when the input is a Series/ndarray

        Return numpy.ndarray, ExtensionArray or NumpyExtensionArray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> pd.unique(pd.Series([2, 1, 3, 3]))
    array([2, 1, 3])

    >>> pd.unique(pd.Series([2] + [1] * 5))
    array([2, 1])

    >>> pd.unique(pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    array(['2016-01-01T00:00:00.000000'], dtype='datetime64[us]')

    >>> pd.unique(
    ...     pd.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ],
    ...         dtype="M8[ns, US/Eastern]",
    ...     )
    ... )
    <DatetimeArray>
    ['2016-01-01 00:00:00-05:00']
    Length: 1, dtype: datetime64[ns, US/Eastern]

    >>> pd.unique(
    ...     pd.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ],
    ...         dtype="M8[ns, US/Eastern]",
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00'],
            dtype='datetime64[ns, US/Eastern]',
            freq=None)

    >>> pd.unique(np.array(list("baabc"), dtype="O"))
    array(['b', 'a', 'c'], dtype=object)

    An unordered Categorical will return categories in the
    order of appearance.

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    ['b', 'a', 'c']
    Categories (3, str): ['a', 'b', 'c']

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    ['b', 'a', 'c']
    Categories (3, str): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(
    ...     pd.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    ['b', 'a', 'c']
    Categories (3, str): ['a' < 'b' < 'c']

    An array of tuples

    >>> pd.unique(pd.Series([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")]).values)
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)

    A NumpyExtensionArray of complex

    >>> pd.unique(pd.array([1 + 1j, 2, 3]))
    <NumpyExtensionArray>
    [(1+1j), (2+0j), (3+0j)]
    Length: 3, dtype: complex128
    """
    return unique_with_mask(values)


def nunique_ints(values: ArrayLike) -> int:
    # [FUNCTION DOC]
    # Counts distinct values of an integer array-like using a bin count. Input:
    # values (assumed integral; this is not checked, and negative values would make
    # the bin count fail). Returns zero for empty input; otherwise the number of
    # non-empty bins as a numpy integer.
    """
    Return the number of unique values for integer array-likes.

    Significantly faster than pandas.unique for long enough sequences.
    No checks are done to ensure input is integral.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    int : The number of unique values in ``values``
    """
    if len(values) == 0:
        return 0
    values = _ensure_data(values)
    # bincount requires intp
    result = (np.bincount(values.ravel().astype("intp")) != 0).sum()
    return result


def unique_with_mask(values, mask: npt.NDArray[np.bool_] | None = None):
    # [FUNCTION DOC]
    # Implementation of the unique function that also supports masked data.
    # Inputs: values (array-like as for unique) and mask (optional boolean array,
    # true marks missing entries, for masked extension arrays). Extension dtypes and
    # Index inputs dispatch to their own unique method. Otherwise the data is
    # coerced, a hashtable is built, and the uniques are computed and converted back
    # to the original dtype. Returns the uniques, or, when a mask was given, a pair
    # of the uniques and the boolean mask of the uniques.
    """See algorithms.unique for docs. Takes a mask for masked arrays."""
    values = _ensure_arraylike(values, func_name="unique")

    if isinstance(values.dtype, ExtensionDtype):
        # Dispatch to extension dtype's unique.
        return values.unique()

    if isinstance(values, ABCIndex):
        # Dispatch to Index's unique.
        return values.unique()

    original = values
    hashtable, values = _get_hashtable_algo(values)

    table = hashtable(len(values))
    if mask is None:
        uniques = table.unique(values)
        uniques = _reconstruct_data(uniques, original.dtype, original)
        return uniques

    else:
        uniques, mask = table.unique(values, mask=mask)
        uniques = _reconstruct_data(uniques, original.dtype, original)
        assert mask is not None  # for mypy
        return uniques, mask.astype("bool")


unique1d = unique


_MINIMUM_COMP_ARR_LEN = 1_000_000


def isin(comps: ListLike, values: ListLike) -> npt.NDArray[np.bool_]:
    # [FUNCTION DOC]
    # Elementwise membership test. Inputs: comps (list-like of elements to test)
    # and values (list-like of allowed values). Raises a type error if either is not
    # list-like (naming the offending type). Plain sequences used as values are
    # converted to arrays, falling back to object arrays in some mixed-type cases to
    # avoid unwanted upcasting to float; MultiIndex values become an array of
    # tuples. Extension array comps use their own method; datetime-like comps
    # dispatch to the datetime array method; datetime-like values against
    # non-object comps give all false; datetime-like or extension-typed values are
    # converted (to object, or to plain arrays) and the test recurses.
    # For numpy data: for very large comps (over one million elements) and few
    # values (at most 26) without the NA singleton, uses numpy membership (taking
    # care of NaN equality when values contain missing); otherwise both sides are
    # cast to a common dtype and the hashtable membership routine is used.
    # Returns a boolean numpy array with the length of comps.
    # KNOWN ISSUE: the comment about not being quite right for sparse and
    # categorical data sits after a return statement and the branch it refers to is
    # unreachable code beyond the return.
    """
    Compute the isin boolean array.

    Parameters
    ----------
    comps : list-like
    values : list-like

    Returns
    -------
    ndarray[bool]
        Same length as `comps`.
    """
    if not is_list_like(comps):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(comps).__name__}`"
        )
    if not is_list_like(values):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a `{type(values).__name__}`"
        )

    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        orig_values = list(values)
        values = _ensure_arraylike(orig_values, func_name="isin-targets")

        if (
            len(values) > 0
            and values.dtype.kind in "iufcb"
            and not is_signed_integer_dtype(comps)
            and not is_dtype_equal(values, comps)
        ):
            # GH#46485 Use object to avoid upcast to float64 later
            # TODO: Share with _find_common_type_compat
            values = construct_1d_object_array_from_listlike(orig_values)

    elif isinstance(values, ABCMultiIndex):
        # Avoid raising in extract_array
        values = np.array(values)
    else:
        values = extract_array(values, extract_numpy=True, extract_range=True)

    comps_array = _ensure_arraylike(comps, func_name="isin")
    comps_array = extract_array(comps_array, extract_numpy=True)
    if not isinstance(comps_array, np.ndarray):
        # i.e. Extension Array
        return comps_array.isin(values)

    elif needs_i8_conversion(comps_array.dtype):
        # Dispatch to DatetimeLikeArrayMixin.isin
        return pd_array(comps_array).isin(values)
    elif needs_i8_conversion(values.dtype) and not is_object_dtype(comps_array.dtype):
        # e.g. comps_array are integers and values are datetime64s
        return np.zeros(comps_array.shape, dtype=bool)
        # TODO: not quite right ... Sparse/Categorical
    elif needs_i8_conversion(values.dtype):
        return isin(comps_array, values.astype(object))

    elif isinstance(values.dtype, ExtensionDtype):
        return isin(np.asarray(comps_array), np.asarray(values))

    # GH16012
    # Ensure np.isin doesn't get object types or it *may* throw an exception
    # Albeit hashmap has O(1) look-up (vs. O(logn) in sorted array),
    # isin is faster for small sizes

    # GH60678
    # Ensure values don't contain <NA>, otherwise it throws exception with np.in1d

    if (
        len(comps_array) > _MINIMUM_COMP_ARR_LEN
        and len(values) <= 26
        and comps_array.dtype != object
        and not any(v is NA for v in values)
    ):
        # If the values include nan we need to check for nan explicitly
        # since np.nan it not equal to np.nan
        if isna(values).any():

            def f(c, v):
                # [FUNCTION DOC]
                # HELPER: membership with NaN awareness; true where the element is
                # in v by numpy membership or where the element itself is NaN.
                return np.logical_or(np.isin(c, v).ravel(), np.isnan(c))

        else:
            f = lambda a, b: np.isin(a, b).ravel()

    else:
        common = np_find_common_type(values.dtype, comps_array.dtype)
        values = values.astype(common, copy=False)
        comps_array = comps_array.astype(common, copy=False)
        f = htable.ismember

    return f(comps_array, values)


def factorize_array(
    values: np.ndarray,
    use_na_sentinel: bool = True,
    size_hint: int | None = None,
    na_value: object = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.intp], np.ndarray]:
    # [FUNCTION DOC]
    # Factorizes a numpy array into integer codes and unique values using a
    # hashtable, without type coercion by the caller. Inputs: values (numpy array),
    # use_na_sentinel (default true: missing values get code minus one and are
    # dropped from the uniques; if false they get ordinary codes and appear in the
    # uniques), size_hint (optional expected number of uniques; default the array
    # length), na_value (optional value treated as missing; for datetime-like data
    # it is replaced by the integer missing sentinel, assuming the caller passed a
    # proper not-a-time) and mask (optional boolean array, true = missing, used
    # instead of na_value/NaN detection).
    # Returns a pair: platform integer codes and the uniques converted back to the
    # original dtype.
    """
    Factorize a numpy array to codes and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    size_hint : int, optional
        Passed through to the hashtable's 'get_labels' method
    na_value : object, optional
        A value in `values` to consider missing. Note: only use this
        parameter when you know that you don't have any values pandas would
        consider missing in the array (NaN for float data, iNaT for
        datetimes, etc.).
    mask : ndarray[bool], optional
        If not None, the mask is used as indicator for missing values
        (True = missing, False = valid) instead of `na_value` or
        condition "val != val".

    Returns
    -------
    codes : ndarray[np.intp]
    uniques : ndarray
    """
    original = values
    if values.dtype.kind in "mM":
        # _get_hashtable_algo will cast dt64/td64 to i8 via _ensure_data, so we
        #  need to do the same to na_value. We are assuming here that the passed
        #  na_value is an appropriately-typed NaT.
        # e.g. test_where_datetimelike_categorical
        na_value = iNaT

    hash_klass, values = _get_hashtable_algo(values)

    table = hash_klass(size_hint or len(values))
    uniques, codes = table.factorize(
        values,
        na_sentinel=-1,
        na_value=na_value,
        mask=mask,
        ignore_na=use_na_sentinel,
    )

    # re-cast e.g. i8->dt64/td64, uint8->bool
    uniques = _reconstruct_data(uniques, original.dtype, original)

    codes = ensure_platform_int(codes)
    return codes, uniques


@set_module("pandas")
def factorize(
    values,
    sort: bool = False,
    use_na_sentinel: bool = True,
    size_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray | Index]:
    # [FUNCTION DOC]
    # Public top-level function: encodes values as integer codes plus the array of
    # distinct values. Inputs: values (one dimensional sequence; a Series or Index
    # dispatches to its own factorize method; other non-array inputs raise a type
    # error), sort (default false; if true the uniques are sorted and the codes
    # remapped accordingly), use_na_sentinel (default true: missing values get
    # code minus one and are excluded from the uniques; if false they receive a
    # normal code and appear in the uniques) and size_hint (optional hashtable size
    # hint, used only for plain numpy input).
    # Behavior: datetime-like arrays with a frequency use a fast path on the array;
    # other extension arrays use their own factorize; numpy arrays are factorized
    # with the hashtable routine (for object arrays with the no-sentinel option,
    # all null flavors are first normalized to the dtype's own missing value
    # without modifying the caller's array). If sort is requested and there are
    # uniques, they are sorted with the safe sort helper. The uniques are then
    # converted back to the original dtype.
    # Returns a pair: codes (integer array) and uniques (numpy array, or an
    # extension array for extension input).
    # Note: the datetime-like frequency fast path does not forward the sentinel
    # option, which is harmless because such arrays cannot contain missing values.
    """
    Encode the object as an enumerated type or categorical variable.

    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. `factorize`
    is available as both a top-level function :func:`pandas.factorize`,
    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

    Parameters
    ----------
    values : sequence
        A 1-D sequence. Sequences that aren't pandas objects are
        coerced to ndarrays before factorization.
    sort : bool, default False
        Sort `uniques` and shuffle `codes` to maintain the
        relationship.
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    size_hint : int, optional
        Hint to the hashtable sizer.

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

    >>> codes, uniques = pd.factorize(np.array(["b", "b", "a", "c", "b"], dtype="O"))
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

    >>> codes, uniques = pd.factorize(np.array(["b", None, "a", "c", "b"], dtype="O"))
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
    # Implementation notes: This method is responsible for 3 things
    # 1.) coercing data to array-like (ndarray, Index, extension array)
    # 2.) factorizing codes and uniques
    # 3.) Maybe boxing the uniques in an Index
    #
    # Step 2 is dispatched to extension types (like Categorical). They are
    # responsible only for factorization. All data coercion, sorting and boxing
    # should happen here.
    if isinstance(values, (ABCIndex, ABCSeries)):
        return values.factorize(sort=sort, use_na_sentinel=use_na_sentinel)

    values = _ensure_arraylike(values, func_name="factorize")
    original = values

    if (
        isinstance(values, (ABCDatetimeArray, ABCTimedeltaArray))
        and values.freq is not None
    ):
        # The presence of 'freq' means we can fast-path sorting and know there
        #  aren't NAs
        codes, uniques = values.factorize(sort=sort)
        return codes, uniques

    elif not isinstance(values, np.ndarray):
        # i.e. ExtensionArray
        codes, uniques = values.factorize(use_na_sentinel=use_na_sentinel)

    else:
        values = np.asarray(values)  # convert DTA/TDA/MultiIndex

        if not use_na_sentinel and values.dtype == object:
            # factorize can now handle differentiating various types of null values.
            # These can only occur when the array has object dtype.
            # However, for backwards compatibility we only use the null for the
            # provided dtype. This may be revisited in the future, see GH#48476.
            null_mask = isna(values)
            if null_mask.any():
                na_value = na_value_for_dtype(values.dtype, compat=False)
                # Don't modify (potentially user-provided) array
                values = np.where(null_mask, na_value, values)

        codes, uniques = factorize_array(
            values,
            use_na_sentinel=use_na_sentinel,
            size_hint=size_hint,
        )

    if sort and len(uniques) > 0:
        uniques, codes = safe_sort(
            uniques,
            codes,
            use_na_sentinel=use_na_sentinel,
            assume_unique=True,
            verify=False,
        )

    uniques = _reconstruct_data(uniques, original.dtype, original)

    return codes, uniques


def value_counts_internal(
    values,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins=None,
    dropna: bool = True,
) -> Series:
    # [FUNCTION DOC]
    # Implementation behind counting distinct values. Inputs: values (array-like,
    # Series or Index), sort (default true: order by count), ascending (default
    # false), normalize (default false: return proportions rather than counts),
    # bins (optional; if given, values are cut into bins first and the intervals are
    # counted; non-numeric data raises a type error saying bins only work with
    # numeric data) and dropna (default true: exclude missing values).
    # Behavior: with bins, counts per interval are computed, missing intervals
    # dropped, ordered by interval, and normalization uses the length of all
    # binned values. For extension arrays the array's own value counts is used; for
    # a MultiIndex a groupby size is used; otherwise the hashtable counting routine
    # is used (half precision keys are widened to single precision) and the keys
    # form an index of the same dtype, preserving the inferred frequency of
    # datetime-like indexes when unsorted and equal. The result is a Series named
    # by count or proportion, indexed by the distinct values (index named after the
    # input's name), sorted with a stable sort if requested, and divided by the
    # total (or the bin denominator) if normalizing.
    from pandas import (
        DatetimeIndex,
        Index,
        Series,
        TimedeltaIndex,
    )

    index_name = getattr(values, "name", None)
    name = "proportion" if normalize else "count"

    if bins is not None:
        from pandas.core.reshape.tile import cut

        if isinstance(values, Series):
            values = values._values

        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError("bins argument only works with numeric data.") from err

        # count, remove nulls (from the index), and but the bins
        result = ii.value_counts(dropna=dropna)
        result.name = name
        result = result[result.index.notna()]
        result.index = result.index.astype("interval")
        result = result.sort_index()

        # if we are dropna and we have NO values
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]

        # normalizing is by len of all (regardless of dropna)
        normalize_denominator = len(ii)

    else:
        normalize_denominator = None
        if is_extension_array_dtype(values):
            # handle Categorical and sparse,
            result = Series(values, copy=False)._values.value_counts(dropna=dropna)
            result.name = name
            result.index.name = index_name

        elif isinstance(values, ABCMultiIndex):
            # GH49558
            levels = list(range(values.nlevels))
            result = (
                Series(index=values, name=name)
                .groupby(level=levels, dropna=dropna)
                .size()
            )
            result.index.names = values.names

        else:
            values = _ensure_arraylike(values, func_name="value_counts")
            keys, counts, _ = value_counts_arraylike(values, dropna)
            if keys.dtype == np.float16:
                keys = keys.astype(np.float32)

            # Starting in 3.0, we no longer perform dtype inference on the
            #  Index object we construct here, xref GH#56161
            idx = Index(keys, dtype=keys.dtype, name=index_name, copy=False)

            if (
                not sort
                and isinstance(values, (DatetimeIndex, TimedeltaIndex))
                and idx.equals(values)
                and values.inferred_freq is not None
            ):
                # Preserve freq of original index
                idx.freq = values.inferred_freq  # type: ignore[attr-defined]

            result = Series(counts, index=idx, name=name, copy=False)

    if sort:
        result = result.sort_values(ascending=ascending, kind="stable")

    if normalize:
        if normalize_denominator is not None:
            result = result / normalize_denominator
        else:
            result = result / result.sum()

    return result


# Called once from SparseArray, otherwise could be private
def value_counts_arraylike(
    values: np.ndarray, dropna: bool, mask: npt.NDArray[np.bool_] | None = None
) -> tuple[ArrayLike, npt.NDArray[np.int64], int]:
    # [FUNCTION DOC]
    # Counts occurrences of each distinct value using a hashtable. Inputs: values
    # (array), dropna (whether to exclude missing values) and mask (optional
    # boolean array, true = missing, for masked data).
    # Behavior: data is coerced for the hashtable; counting yields keys, counts and
    # the number of missing entries seen. For datetime-like dtypes with dropna, the
    # entries whose key is the integer missing sentinel are removed. Keys are
    # converted back to the original dtype.
    # Returns a triple: the keys, the 64-bit counts and the missing-entry counter.
    # KNOWN ISSUE: the docstring lists only two return values, but a third (the
    # missing count) is returned.
    # Called once from SparseArray, otherwise could be private (per a comment).
    """
    Parameters
    ----------
    values : np.ndarray
    dropna : bool
    mask : np.ndarray[bool] or None, default None

    Returns
    -------
    uniques : np.ndarray
    counts : np.ndarray[np.int64]
    """
    original = values
    values = _ensure_data(values)

    keys, counts, na_counter = htable.value_count(values, dropna, mask=mask)

    if needs_i8_conversion(original.dtype):
        # datetime, timedelta, or period

        if dropna:
            mask = keys != iNaT
            keys, counts = keys[mask], counts[mask]

    res_keys = _reconstruct_data(keys, original.dtype, original)
    return res_keys, counts, na_counter


def duplicated(
    values: ArrayLike,
    keep: Literal["first", "last", False] = "first",
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    # [FUNCTION DOC]
    # Marks duplicated entries. Inputs: values (numpy or extension array; coerced to
    # a hashtable-friendly dtype), keep (first, last, or false; default first:
    # which occurrence is NOT marked, false marks all duplicates) and mask
    # (optional boolean array of elements to exclude from the check). Returns a
    # boolean array of the same length.
    """
    Return boolean ndarray denoting duplicate values.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        Array over which to check for duplicate values.
    keep : {'first', 'last', False}, default 'first'
        - ``first`` : Mark duplicates as ``True`` except for the first
          occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the last
          occurrence.
        - False : Mark all duplicates as ``True``.
    mask : ndarray[bool], optional
        array indicating which elements to exclude from checking

    Returns
    -------
    duplicated : ndarray[bool]
    """
    values = _ensure_data(values)
    return htable.duplicated(values, keep=keep, mask=mask)


def mode(
    values: ArrayLike, dropna: bool = True, mask: npt.NDArray[np.bool_] | None = None
) -> tuple[np.ndarray, npt.NDArray[np.bool_]] | ExtensionArray:
    # [FUNCTION DOC]
    # Computes the most frequent value(s). Inputs: values (array-like; other types
    # raise a type error), dropna (default true: do not count missing values) and
    # mask (optional boolean array of missing entries for masked data).
    # Behavior: datetime-like data is wrapped and dispatched to the array's own
    # mode method, whose result (an extension array) is returned directly. Other
    # data is coerced and the hashtable mode routine run; if it returns a result
    # mask (masked input) the pair is returned immediately without sorting or
    # type restoration. Otherwise an all-false mask is created, the modes are
    # sorted with the safe sort helper (a failure to sort emits a warning that the
    # modes could not be sorted, attributed to the caller, and leaves them
    # unsorted), and the result is converted back to the original dtype.
    # Returns a pair of modes and a boolean mask, except for datetime-like input.
    # KNOWN ISSUE: the docstring omits the mask parameter and claims the return type
    # is a union where the pair is returned in almost all cases.
    """
    Returns the mode(s) of an array.

    Parameters
    ----------
    values : array-like
        Array over which to check for duplicate values.
    dropna : bool, default True
        Don't consider counts of NaN/NaT.

    Returns
    -------
    Union[Tuple[np.ndarray, npt.NDArray[np.bool_]], ExtensionArray]
    """
    values = _ensure_arraylike(values, func_name="mode")
    original = values

    if needs_i8_conversion(values.dtype):
        # Got here with ndarray; dispatch to DatetimeArray/TimedeltaArray.
        values = ensure_wrapped_if_datetimelike(values)
        values = cast("ExtensionArray", values)
        return values._mode(dropna=dropna)

    values = _ensure_data(values)

    npresult, res_mask = htable.mode(values, dropna=dropna, mask=mask)
    if res_mask is None:
        res_mask = np.zeros(npresult.shape, dtype=np.bool_)
    else:
        return npresult, res_mask

    try:
        npresult = safe_sort(npresult)
    except TypeError as err:
        warnings.warn(
            f"Unable to sort modes: {err}",
            stacklevel=find_stack_level(),
        )

    result = _reconstruct_data(npresult, original.dtype, original)
    return result, res_mask


def rank(
    values: ArrayLike,
    axis: AxisInt = 0,
    method: str = "average",
    na_option: str = "keep",
    ascending: bool = True,
    pct: bool = False,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.float64]:
    # [FUNCTION DOC]
    # Ranks values along an axis. Inputs: values (array of one or two dimensions),
    # axis (default 0; used for 2D only), method (tie-breaking: average, min, max,
    # first, dense; default average), na_option (keep or top; default keep),
    # ascending (default true), pct (default false: return percentile ranks) and
    # mask (optional boolean array of entries to exclude; only supported for 1D,
    # an assertion error is raised for 2D input with a mask).
    # Datetime-like data is flagged and converted to integers. Dispatches to the
    # one or two dimensional compiled rank routine. Raises a type error for more
    # than two dimensions. Returns a float array of ranks.
    # KNOWN ISSUE: the docstring has no Returns section, and describes na_option as
    # only keep or top while the compiled routine may accept other values (such as
    # bottom).
    """
    Rank the values along a given axis.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
        Array whose values will be ranked. The number of dimensions in this
        array must not exceed 2.
    axis : int, default 0
        Axis over which to perform rankings.
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        The method by which tiebreaks are broken during the ranking.
    na_option : {'keep', 'top'}, default 'keep'
        The method by which NaNs are placed in the ranking.
        - ``keep``: rank each NaN value with a NaN ranking
        - ``top``: replace each NaN with either +/- inf so that they
                   there are ranked at the top
    ascending : bool, default True
        Whether or not the elements should be ranked in ascending order.
    pct : bool, default False
        Whether or not to the display the returned rankings in integer form
        (e.g. 1, 2, 3) or in percentile form (e.g. 0.333..., 0.666..., 1).
    mask : bool ndarray, optional
        Boolean array indicating which elements to exclude from ranking.
    """
    is_datetimelike = needs_i8_conversion(values.dtype)
    values = _ensure_data(values)

    if values.ndim == 1:
        ranks = algos.rank_1d(
            values,
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
            mask=mask,
        )
    elif values.ndim == 2:
        assert mask is None
        ranks = algos.rank_2d(
            values,
            axis=axis,
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
    else:
        raise TypeError("Array with ndim > 2 are not supported.")

    return ranks


# ---- #
# take #
# ---- #


@set_module("pandas.api.extensions")
def take(
    arr,
    indices: TakeIndexer,
    axis: AxisInt = 0,
    allow_fill: bool = False,
    fill_value=None,
):
    # [FUNCTION DOC]
    # Public function (exposed in the extension API) that takes elements by
    # position. Inputs: arr (numpy array, extension array, Index, Series or numpy
    # wrapper array; otherwise a type error), indices (integer sequence or array),
    # axis (default 0), allow_fill (default false) and fill_value (default none,
    # meaning the dtype's missing value; only used if allow_fill).
    # Behavior: indices are converted to platform integers. With allow_fill false,
    # the object's own take is used and negative indices count from the end (numpy
    # style). With allow_fill true, the indices are validated against the axis
    # length (negative values other than minus one raise a value error, out of
    # bounds raises an index error) and the low-level take with fill is used, in
    # which minus one produces the fill value.
    # Returns an array of the same kind as the input.
    """
    Take elements from an array.

    Parameters
    ----------
    arr : numpy.ndarray, ExtensionArray, Index, or Series
        Input array.
    indices : sequence of int or one-dimensional np.ndarray of int
        Indices to be taken.
    axis : int, default 0
        The axis over which to select values.
    allow_fill : bool, default False
        How to handle negative values in `indices`.

        * False: negative values in `indices` indicate positional indices
          from the right (the default). This is similar to :func:`numpy.take`.

        * True: negative values in `indices` indicate
          missing values. These values are set to `fill_value`. Any other
          negative values raise a ``ValueError``.

    fill_value : any, optional
        Fill value to use for NA-indices when `allow_fill` is True.
        This may be ``None``, in which case the default NA value for
        the type (``self.dtype.na_value``) is used.

        For multi-dimensional `arr`, each *element* is filled with
        `fill_value`.

    Returns
    -------
    ndarray or ExtensionArray
        Same type as the input.

    Raises
    ------
    IndexError
        When `indices` is out of bounds for the array.
    ValueError
        When the indexer contains negative values other than ``-1``
        and `allow_fill` is True.

    Notes
    -----
    When `allow_fill` is False, `indices` may be whatever dimensionality
    is accepted by NumPy for `arr`.

    When `allow_fill` is True, `indices` should be 1-D.

    See Also
    --------
    numpy.take : Take elements from an array along an axis.

    Examples
    --------
    >>> import pandas as pd

    With the default ``allow_fill=False``, negative numbers indicate
    positional indices from the right.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1])
    array([10, 10, 30])

    Setting ``allow_fill=True`` will place `fill_value` in those positions.

    >>> pd.api.extensions.take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
    array([10., 10., nan])

    >>> pd.api.extensions.take(
    ...     np.array([10, 20, 30]), [0, 0, -1], allow_fill=True, fill_value=-10
    ... )
    array([ 10,  10, -10])
    """
    if not isinstance(
        arr,
        (np.ndarray, ABCExtensionArray, ABCIndex, ABCSeries, ABCNumpyExtensionArray),
    ):
        # GH#52981
        raise TypeError(
            "pd.api.extensions.take requires a numpy.ndarray, ExtensionArray, "
            f"Index, Series, or NumpyExtensionArray got {type(arr).__name__}."
        )

    indices = ensure_platform_int(indices)

    if allow_fill:
        # Pandas style, -1 means NA
        validate_indices(indices, arr.shape[axis])
        # error: Argument 1 to "take_nd" has incompatible type
        # "ndarray[Any, Any] | ExtensionArray | Index | Series"; expected
        # "ndarray[Any, Any]"
        result = take_nd(
            arr,  # type: ignore[arg-type]
            indices,
            axis=axis,
            allow_fill=True,
            fill_value=fill_value,
        )
    else:
        # NumPy style
        # error: Unexpected keyword argument "axis" for "take" of "ExtensionArray"
        result = arr.take(indices, axis=axis)  # type: ignore[call-arg,assignment]
    return result


# ------------ #
# searchsorted #
# ------------ #


def searchsorted(
    arr: ArrayLike,
    value: NumpyValueArrayLike | ExtensionArray,
    side: Literal["left", "right"] = "left",
    sorter: NumpySorter | None = None,
) -> npt.NDArray[np.intp] | np.intp:
    # [FUNCTION DOC]
    # Finds insertion positions that keep a sorted array sorted. Inputs: arr (numpy
    # array, extension array or Series; ascending unless a sorter is given), value
    # (scalar or array-like to insert), side (left or right; default left) and
    # sorter (optional integer indexer that sorts arr; converted to platform
    # integers).
    # Behavior: for integer numpy arrays with integer values, the value is cast to
    # the array's dtype (to avoid slow recasting of the array by numpy) unless it
    # would overflow, in which case the value's own dtype is used. Otherwise
    # datetime-like arrays are wrapped so that Timestamp-like values compare
    # properly. The search is then delegated to the array's own method.
    # Returns an integer for scalar value, else an integer array.
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `arr` (a) such that, if the
    corresponding elements in `value` were inserted before the indices,
    the order of `arr` would be preserved.

    Assuming that `arr` is sorted:

    ======  ================================
    `side`  returned index `i` satisfies
    ======  ================================
    left    ``arr[i-1] < value <= self[i]``
    right   ``arr[i-1] <= value < self[i]``
    ======  ================================

    Parameters
    ----------
    arr: np.ndarray, ExtensionArray, Series
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    value : array-like or scalar
        Values to insert into `arr`.
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
    """
    if sorter is not None:
        sorter = ensure_platform_int(sorter)

    if (
        isinstance(arr, np.ndarray)
        and arr.dtype.kind in "iu"
        and (is_integer(value) or is_integer_dtype(value))
    ):
        # if `arr` and `value` have different dtypes, `arr` would be
        # recast by numpy, causing a slow search.
        # Before searching below, we therefore try to give `value` the
        # same dtype as `arr`, while guarding against integer overflows.
        iinfo = np.iinfo(arr.dtype.type)
        value_arr = np.array([value]) if is_integer(value) else np.array(value)
        if (value_arr >= iinfo.min).all() and (value_arr <= iinfo.max).all():
            # value within bounds, so no overflow, so can convert value dtype
            # to dtype of arr
            dtype = arr.dtype
        else:
            dtype = value_arr.dtype

        if is_integer(value):
            # We know that value is int
            value = cast(int, dtype.type(value))
        else:
            value = pd_array(cast(ArrayLike, value), dtype=dtype)
    else:
        # E.g. if `arr` is an array with dtype='datetime64[ns]'
        # and `value` is a pd.Timestamp, we may need to convert value
        arr = ensure_wrapped_if_datetimelike(arr)

    # Argument 1 to "searchsorted" of "ndarray" has incompatible type
    # "Union[NumpyValueArrayLike, ExtensionArray]"; expected "NumpyValueArrayLike"
    return arr.searchsorted(value, side=side, sorter=sorter)  # type: ignore[arg-type]


# ---- #
# diff #
# ---- #

_diff_special = {"float64", "float32", "int64", "int32", "int16", "int8"}


def diff(arr, n: int | float | np.integer | np.floating, axis: AxisInt = 0):
    # [FUNCTION DOC]
    # Computes the difference between each element and the element n positions
    # earlier along an axis (the equivalent of subtracting a shifted copy); for
    # booleans the exclusive-or is used instead of subtraction.
    # Inputs: arr (numpy array or extension array), n (number of periods; an
    # integer or a float with an integer value; any other value raises a value
    # error saying periods must be an integer; negative values look forward) and
    # axis (default 0; for one dimensional input it is effectively ignored).
    # Behavior: numpy-wrapper extension arrays are converted to numpy first.
    # Other extension arrays must support the needed operator (else a type error
    # says there is no diff method) and only axis zero (else a value error); the
    # result is the operator applied to the array and its shifted copy. For numpy
    # data: datetime-like values are processed as integers using the integer missing
    # sentinel and the result is viewed as timedelta; booleans produce an object
    # result; small integers produce single-precision floats and other integers
    # double floats (so NaN can be held). The first (or last) n positions along the
    # axis are filled with the missing value. Common dtypes use a compiled routine;
    # others use slicing arithmetic. Returns a new array of the same shape (for 2D
    # input; 1D input gives 1D output).
    # KNOWN ISSUE: the docstring documents a parameter called stacklevel that the
    # function does not accept, and describes n as an int only.
    """
    difference of n between self,
    analogous to s-s.shift(n)

    Parameters
    ----------
    arr : ndarray or ExtensionArray
    n : int
        number of periods
    axis : {0, 1}
        axis to shift on
    stacklevel : int, default 3
        The stacklevel for the lost dtype warning.

    Returns
    -------
    shifted
    """

    # added a check on the integer value of period
    # see https://github.com/pandas-dev/pandas/issues/56607
    if not lib.is_integer(n):
        if not (is_float(n) and n.is_integer()):
            raise ValueError("periods must be an integer")
        n = int(n)
    na = np.nan
    dtype = arr.dtype

    is_bool = is_bool_dtype(dtype)
    if is_bool:
        op = operator.xor
    else:
        op = operator.sub

    if isinstance(dtype, NumpyEADtype):
        # NumpyExtensionArray cannot necessarily hold shifted versions of itself.
        arr = arr.to_numpy()
        dtype = arr.dtype

    if not isinstance(arr, np.ndarray):
        # i.e ExtensionArray
        if hasattr(arr, f"__{op.__name__}__"):
            if axis != 0:
                raise ValueError(f"cannot diff {type(arr).__name__} on axis={axis}")
            return op(arr, arr.shift(n))
        else:
            raise TypeError(
                f"{type(arr).__name__} has no 'diff' method. "
                "Convert to a suitable dtype prior to calling 'diff'."
            )

    is_timedelta = False
    if arr.dtype.kind in "mM":
        dtype = np.int64
        arr = arr.view("i8")
        na = iNaT
        is_timedelta = True

    elif is_bool:
        # We have to cast in order to be able to hold np.nan
        dtype = np.object_

    elif dtype.kind in "iu":
        # We have to cast in order to be able to hold np.nan

        # int8, int16 are incompatible with float64,
        # see https://github.com/cython/cython/issues/2646
        if arr.dtype.name in ["int8", "int16"]:
            dtype = np.float32
        else:
            dtype = np.float64

    orig_ndim = arr.ndim
    if orig_ndim == 1:
        # reshape so we can always use algos.diff_2d
        arr = arr.reshape(-1, 1)
        # TODO: require axis == 0

    dtype = np.dtype(dtype)
    out_arr = np.empty(arr.shape, dtype=dtype)

    na_indexer = [slice(None)] * 2
    na_indexer[axis] = slice(None, n) if n >= 0 else slice(n, None)
    out_arr[tuple(na_indexer)] = na

    if arr.dtype.name in _diff_special:
        # TODO: can diff_2d dtype specialization troubles be fixed by defining
        #  out_arr inside diff_2d?
        algos.diff_2d(arr, out_arr, int(n), axis, datetimelike=is_timedelta)
    else:
        # To keep mypy happy, _res_indexer is a list while res_indexer is
        #  a tuple, ditto for lag_indexer.
        _res_indexer = [slice(None)] * 2
        _res_indexer[axis] = slice(n, None) if n >= 0 else slice(None, n)
        res_indexer = tuple(_res_indexer)

        _lag_indexer = [slice(None)] * 2
        _lag_indexer[axis] = slice(None, -n) if n > 0 else slice(-n, None)
        lag_indexer = tuple(_lag_indexer)

        out_arr[res_indexer] = op(arr[res_indexer], arr[lag_indexer])

    if is_timedelta:
        out_arr = out_arr.view("timedelta64[ns]")

    if orig_ndim == 1:
        out_arr = out_arr[:, 0]
    return out_arr


# --------------------------------------------------------------------
# Helper functions


# Note: safe_sort is in algorithms.py instead of sorting.py because it is
#  low-dependency, is used in this module, and used private methods from
#  this module.
def safe_sort(
    values: Index | ArrayLike,
    codes: npt.NDArray[np.intp] | None = None,
    use_na_sentinel: bool = True,
    assume_unique: bool = False,
    verify: bool = True,
) -> AnyArrayLike | tuple[AnyArrayLike, np.ndarray]:
    # [FUNCTION DOC]
    # Sorts values robustly (supports mixed int/str, tuples, and missing values) and
    # optionally remaps integer codes that refer to positions in the values.
    # Inputs: values (numpy array, extension array or Index; anything else raises a
    # type error), codes (optional integer array of positions into values; if given
    # it must be list-like else a type error, and values must be unique else a value
    # error unless assume_unique), use_na_sentinel (default true: minus one in
    # codes is preserved as missing; if false, codes are remapped with wrap-around
    # indexing), assume_unique (default false; skips the uniqueness check) and
    # verify (default true; out-of-range codes are first replaced by minus one, which
    # mutates the codes array produced from the input).
    # Behavior: integer-and-string mixtures are ordered by a dedicated helper
    # (numbers, then strings, then nulls); otherwise the values are sorted normally,
    # falling back to the tuple sorter for tuple elements or to the mixed sorter
    # when ordinary sorting fails.
    # Returns the sorted values alone when codes is none, otherwise a pair of sorted
    # values and the remapped platform-integer codes.
    # KNOWN ISSUE: the verification step modifies the caller's codes array in place
    # when it is already a platform-integer numpy array (no copy is made).
    """
    Sort ``values`` and reorder corresponding ``codes``.

    ``values`` should be unique if ``codes`` is not None.
    Safe for use with mixed types (int, str), orders ints before strs.

    Parameters
    ----------
    values : list-like
        Sequence; must be unique if ``codes`` is not None.
    codes : np.ndarray[intp] or None, default None
        Indices to ``values``. All out of bound indices are treated as
        "not found" and will be masked with ``-1``.
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    assume_unique : bool, default False
        When True, ``values`` are assumed to be unique, which can speed up
        the calculation. Ignored when ``codes`` is None.
    verify : bool, default True
        Check if codes are out of bound for the values and put out of bound
        codes equal to ``-1``. If ``verify=False``, it is assumed there
        are no out of bound codes. Ignored when ``codes`` is None.

    Returns
    -------
    ordered : AnyArrayLike
        Sorted ``values``
    new_codes : ndarray
        Reordered ``codes``; returned when ``codes`` is not None.

    Raises
    ------
    TypeError
        * If ``values`` is not list-like or if ``codes`` is neither None
        nor list-like
        * If ``values`` cannot be sorted
    ValueError
        * If ``codes`` is not None and ``values`` contain duplicates.
    """
    if not isinstance(values, (np.ndarray, ABCExtensionArray, ABCIndex)):
        raise TypeError(
            "Only np.ndarray, ExtensionArray, and Index objects are allowed to "
            "be passed to safe_sort as values"
        )

    sorter = None
    ordered: AnyArrayLike

    if (
        not isinstance(values.dtype, ExtensionDtype)
        and lib.infer_dtype(values, skipna=False) == "mixed-integer"
    ):
        ordered = _sort_mixed(values)
    else:
        try:
            sorter = values.argsort()
            ordered = values.take(sorter)
        except (TypeError, decimal.InvalidOperation):
            # Previous sorters failed or were not applicable, try `_sort_mixed`
            # which would work, but which fails for special case of 1d arrays
            # with tuples.
            if values.size and isinstance(values[0], tuple):
                # error: Argument 1 to "_sort_tuples" has incompatible type
                # "Union[Index, ExtensionArray, ndarray[Any, Any]]"; expected
                # "ndarray[Any, Any]"
                ordered = _sort_tuples(values)  # type: ignore[arg-type]
            else:
                ordered = _sort_mixed(values)

    # codes:

    if codes is None:
        return ordered

    if not is_list_like(codes):
        raise TypeError(
            "Only list-like objects or None are allowed to "
            "be passed to safe_sort as codes"
        )
    codes = ensure_platform_int(np.asarray(codes))

    if not assume_unique and not len(unique(values)) == len(values):
        raise ValueError("values should be unique if codes is not None")

    if sorter is None:
        # mixed types
        # error: Argument 1 to "_get_hashtable_algo" has incompatible type
        # "Union[Index, ExtensionArray, ndarray[Any, Any]]"; expected
        # "ndarray[Any, Any]"
        hash_klass, values = _get_hashtable_algo(values)  # type: ignore[arg-type]
        t = hash_klass(len(values))
        t.map_locations(values)
        # error: Argument 1 to "lookup" of "HashTable" has incompatible type
        # "ExtensionArray | ndarray[Any, Any] | Index | Series"; expected "ndarray"
        sorter = ensure_platform_int(t.lookup(ordered))  # type: ignore[arg-type]

    if use_na_sentinel:
        # take_nd is faster, but only works for na_sentinels of -1
        order2 = sorter.argsort()
        if verify:
            mask = (codes < -len(values)) | (codes >= len(values))
            codes[mask] = -1
        new_codes = take_nd(order2, codes, fill_value=-1)
    else:
        reverse_indexer = np.empty(len(sorter), dtype=int)
        reverse_indexer.put(sorter, np.arange(len(sorter)))
        # Out of bound indices will be masked with `-1` next, so we
        # may deal with them here without performance loss using `mode='wrap'`
        new_codes = reverse_indexer.take(codes, mode="wrap")

    return ordered, ensure_platform_int(new_codes)


def _sort_mixed(values) -> AnyArrayLike:
    # [FUNCTION DOC]
    # HELPER: sorting for one dimensional data of mixed types, used by the safe sort.
    # Input: values (array-like supporting boolean indexing and take). Partitions
    # entries into strings, nulls and everything else (assumed numeric), sorts the
    # strings and the numbers separately, and returns the values taken in the order
    # numbers, strings, nulls (nulls in original order).
    """order ints before strings before nulls in 1d arrays"""
    str_pos = np.array([isinstance(x, str) for x in values], dtype=bool)
    null_pos = np.array([isna(x) for x in values], dtype=bool)
    num_pos = ~str_pos & ~null_pos
    str_argsort = np.argsort(values[str_pos])
    num_argsort = np.argsort(values[num_pos])
    # convert boolean arrays to positional indices, then order by underlying values
    str_locs = str_pos.nonzero()[0].take(str_argsort)
    num_locs = num_pos.nonzero()[0].take(num_argsort)
    null_locs = null_pos.nonzero()[0]
    locs = np.concatenate([num_locs, str_locs, null_locs])
    return values.take(locs)


def _sort_tuples(values: np.ndarray) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: sorts a one dimensional array of tuples lexicographically, used by the
    # safe sort. Input: values (array of tuples). The tuples are split into one
    # array per position (so each column keeps its own types and missing values),
    # a lexicographic sorting indexer over those columns is computed (ascending) and
    # the original values are reordered by it. Returns the sorted array. The
    # imports of the conversion and sorting helpers are deferred to call time.
    """
    Convert array of tuples (1d) to array of arrays (2d).
    We need to keep the columns separately as they contain different types and
    nans (can't use `np.sort` as it may fail when str and nan are mixed in a
    column as types cannot be compared).
    """
    from pandas.core.internals.construction import to_arrays
    from pandas.core.sorting import lexsort_indexer

    arrays, _ = to_arrays(values, None)
    indexer = lexsort_indexer(arrays, orders=True)
    return values[indexer]


def union_with_duplicates(
    lvals: ArrayLike | Index, rvals: ArrayLike | Index
) -> ArrayLike | Index:
    # [FUNCTION DOC]
    # Computes a multiset union of two arrays. Inputs: lvals and rvals (numpy or
    # extension arrays or Index objects of identical dtype; the caller guarantees
    # this). For each distinct value (missing values counted too) the result holds
    # as many copies as the larger of its counts in the two inputs. The distinct
    # values come from the combined values in order of first appearance (left
    # first); for two MultiIndex inputs, the unique entries of their concatenation.
    # Returns the repeated unique values, unsorted, as an array (or Index for
    # MultiIndex input).
    """
    Extracts the union from lvals and rvals with respect to duplicates and nans in
    both arrays.

    Parameters
    ----------
    lvals: np.ndarray or ExtensionArray
        left values which is ordered in front.
    rvals: np.ndarray or ExtensionArray
        right values ordered after lvals.

    Returns
    -------
    np.ndarray or ExtensionArray
        Containing the unsorted union of both arrays.

    Notes
    -----
    Caller is responsible for ensuring lvals.dtype == rvals.dtype.
    """
    from pandas import Series

    l_count = value_counts_internal(lvals, dropna=False)
    r_count = value_counts_internal(rvals, dropna=False)
    l_count, r_count = l_count.align(r_count, fill_value=0)
    final_count = np.maximum(l_count.values, r_count.values)
    final_count = Series(final_count, index=l_count.index, dtype="int", copy=False)
    if isinstance(lvals, ABCMultiIndex) and isinstance(rvals, ABCMultiIndex):
        unique_vals = lvals.append(rvals).unique()
    else:
        if isinstance(lvals, ABCIndex):
            lvals = lvals._values
        if isinstance(rvals, ABCIndex):
            rvals = rvals._values
        # error: List item 0 has incompatible type "Union[ExtensionArray,
        # ndarray[Any, Any], Index]"; expected "Union[ExtensionArray,
        # ndarray[Any, Any]]"
        combined = concat_compat([lvals, rvals])  # type: ignore[list-item]
        unique_vals = unique(combined)
        unique_vals = ensure_wrapped_if_datetimelike(unique_vals)
    repeats = final_count.reindex(unique_vals).values
    return np.repeat(unique_vals, repeats)


def map_array(
    arr: ArrayLike,
    mapper,
    na_action: Literal["ignore"] | None = None,
) -> np.ndarray | ExtensionArray | Index:
    # [FUNCTION DOC]
    # Maps each element of an array through a function, dictionary or Series.
    # Inputs: arr (array-like to map; the docstring below omits it), mapper
    # (callable, dict, dict subclass with a default-value hook, or Series) and
    # na_action (none or the string ignore; any other value raises a value error).
    # Behavior: dictionaries with a missing-key hook become a lookup function that
    # normalizes NaN keys; other dict-likes become a Series (an empty mapping gives a
    # float Series so the result is all missing floats; tuple keys are kept as
    # tuples). With a Series mapper (optionally dropping null index entries first
    # when ignoring missing), the result is a lookup of each element in the mapper
    # index, with missing keys giving the missing value. An empty array returns a
    # copy. Otherwise the elements are converted to objects and the function is
    # applied elementwise; with ignore, missing elements are skipped and kept.
    # Returns an array (possibly an Index or extension array if the lookup produced
    # one).
    """
    Map values using an input mapping or function.

    Parameters
    ----------
    mapper : function, dict, or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}, default None
        If 'ignore', propagate NA values, without passing them to the
        mapping correspondence.

    Returns
    -------
    Union[ndarray, Index, ExtensionArray]
        The output of the mapping function applied to the array.
        If the function returns a tuple with more than one element
        a MultiIndex will be returned.
    """
    from pandas import Index

    if na_action not in (None, "ignore"):
        msg = f"na_action must either be 'ignore' or None, {na_action} was passed"
        raise ValueError(msg)

    # we can fastpath dict/Series to an efficient map
    # as we know that we are not going to have to yield
    # python types
    if is_dict_like(mapper):
        if isinstance(mapper, dict) and hasattr(mapper, "__missing__"):
            # If a dictionary subclass defines a default value method,
            # convert mapper to a lookup function (GH #15999).
            dict_with_default = mapper
            mapper = lambda x: dict_with_default[
                np.nan if isinstance(x, float) and np.isnan(x) else x
            ]
        else:
            # Dictionary does not have a default. Thus it's safe to
            # convert to a Series for efficiency.
            # we specify the keys here to handle the
            # possibility that they are tuples

            # The return value of mapping with an empty mapper is
            # expected to be pd.Series(np.nan, ...). As np.nan is
            # of dtype float64 the return value of this method should
            # be float64 as well
            from pandas import Series

            if len(mapper) == 0:
                mapper = Series(mapper, dtype=np.float64)
            elif isinstance(mapper, dict):
                mapper = Series(
                    mapper.values(), index=Index(mapper.keys(), tupleize_cols=False)
                )
            else:
                mapper = Series(mapper)

    if isinstance(mapper, ABCSeries):
        if na_action == "ignore":
            mapper = mapper[mapper.index.notna()]

        # Since values were input this means we came from either
        # a dict or a series and mapper should be an index
        indexer = mapper.index.get_indexer(arr)
        new_values = take_nd(mapper._values, indexer)

        return new_values

    if not len(arr):
        return arr.copy()

    # we must convert to python types
    values = arr.astype(object, copy=False)
    if na_action is None:
        return lib.map_infer(values, mapper)
    else:
        return lib.map_infer_mask(values, mapper, mask=isna(values).view(np.uint8))
