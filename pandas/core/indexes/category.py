# [MODULE DOC]
# Defines the public CategoricalIndex class, an Index whose values are stored
# as a Categorical (integer codes plus a categories Index). Public content:
# CategoricalIndex only. It subclasses the array-backed extension index base
# from the extension module and builds on the generic Index base class; the
# data container and the membership helper come from the categorical array
# module. Many Categorical attributes and methods are attached to the class
# by the name-inheriting class decorators applied at import time (delegating
# properties/methods, some wrapped so they return a new index). The class is
# registered for display under the top-level pandas namespace via a module
# decorator. No other import-time side effects.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
)

import numpy as np

from pandas._libs import index as libindex
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)

from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
)

from pandas.core.arrays.categorical import (
    Categorical,
    contains,
)
from pandas.core.construction import extract_array
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.indexes.extension import (
    NDArrayBackedExtensionIndex,
    inherit_names,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        npt,
    )


@inherit_names(
    [
        "argsort",
        "tolist",
        "codes",
        "categories",
        "ordered",
        "_reverse_indexer",
        "searchsorted",
        "min",
        "max",
    ],
    Categorical,
)
@inherit_names(
    [
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    Categorical,
    wrap=True,
)
@set_module("pandas")
class CategoricalIndex(NDArrayBackedExtensionIndex):
    # [CLASS DOC]
    # Immutable one-dimensional index whose values are categorical. Subclass
    # of the array-backed extension index; the underlying data (both the data
    # and values attributes) is always a Categorical, exposing integer codes,
    # the categories Index and an ordered flag. Lookups use an integer engine
    # chosen to match the codes dtype. Invariant: values not among the
    # categories are stored as missing (code -1). Key members: construction
    # via the new method, equality, containment, reindex (limited), map, and
    # category-manipulation methods inherited from Categorical by decorators.
    """
    Index based on an underlying :class:`Categorical`.

    CategoricalIndex, like Categorical, can only take on a limited,
    and usually fixed, number of possible values (`categories`). Also,
    like Categorical, it might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If `categories` are given, values not in
        `categories` will be replaced with NaN.
    categories : index-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in `dtype`), they
        will be inferred from the `data`.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered
        categorical. If not given here or in `dtype`, the resulting
        categorical will be unordered.
    dtype : CategoricalDtype or "category", optional
        If :class:`CategoricalDtype`, cannot be used together with
        `categories` or `ordered`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    codes
    categories
    ordered

    Methods
    -------
    rename_categories
    reorder_categories
    add_categories
    remove_categories
    remove_unused_categories
    set_categories
    as_ordered
    as_unordered
    map

    Raises
    ------
    ValueError
        If the categories do not validate.
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    Index : The base pandas Index type.
    Categorical : A categorical array.
    CategoricalDtype : Type for categorical data.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#categoricalindex>`__
    for more.

    Examples
    --------
    >>> pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    ``CategoricalIndex`` can also be instantiated from a ``Categorical``:

    >>> c = pd.Categorical(["a", "b", "c", "a", "b", "c"])
    >>> pd.CategoricalIndex(c)
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    Ordered ``CategoricalIndex`` can have a min and max value.

    >>> ci = pd.CategoricalIndex(
    ...     ["a", "b", "c", "a", "b", "c"], ordered=True, categories=["c", "b", "a"]
    ... )
    >>> ci
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['c', 'b', 'a'], ordered=True, dtype='category')
    >>> ci.min()
    'c'
    """

    _typ = "categoricalindex"
    _data_cls = Categorical

    @property
    def _can_hold_strings(self):
        # [FUNCTION DOC]
        # HELPER: property used by generic Index string-handling logic.
        # Returns whether the index can hold strings, delegating to the
        # categories index (so the answer depends on the category dtype).
        return self.categories._can_hold_strings

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached property for integer-key indexing semantics.
        # Returns a bool: whether integer keys given to getitem-style lookups
        # should be treated as positions. Delegates to the categories index,
        # so it is true when the categories are not integer-like.
        return self.categories._should_fallback_to_positional

    codes: np.ndarray
    categories: Index
    ordered: bool | None
    _data: Categorical
    _values: Categorical

    @property
    def _engine_type(self) -> type[libindex.IndexEngine]:
        # [FUNCTION DOC]
        # HELPER: property telling the base Index which hash-engine class to
        # instantiate. Returns the signed-integer engine class (8, 16, 32 or
        # 64 bit) matching the dtype of the codes array. Raises a key error
        # if the codes dtype is not one of those four integer types.
        # self.codes can have dtype int8, int16, int32 or int64, so we need
        # to return the corresponding engine type (libindex.Int8Engine, etc.).
        return {
            np.int8: libindex.Int8Engine,
            np.int16: libindex.Int16Engine,
            np.int32: libindex.Int32Engine,
            np.int64: libindex.Int64Engine,
        }[self.codes.dtype.type]

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Constructor. Parameters: data is a one-dimensional array-like (or
        # Categorical/Series/Index) of values, default none; categories is an
        # optional index-like of unique categories, default none (inferred
        # from data); ordered is an optional bool, default none (unordered
        # unless dtype says otherwise); dtype is an optional categorical
        # dtype or the string category, which cannot be combined with
        # categories/ordered; copy is a bool, default false, whether to copy
        # the input; name is an optional hashable label, default none (taken
        # from data if it has a name). Returns a new CategoricalIndex built
        # from a Categorical. Raises a type error (via the scalar-data
        # helper) when data is a scalar, including none; the Categorical
        # constructor may raise value or type errors for invalid categories.
        name = maybe_extract_name(name, data, cls)

        if is_scalar(data):
            # GH#38944 include None here, which pre-2.0 subbed in []
            cls._raise_scalar_data_error(data)

        data = Categorical(
            data, categories=categories, ordered=ordered, dtype=dtype, copy=copy
        )

        return cls._simple_new(data, name=name)

    # --------------------------------------------------------------------

    def _is_dtype_compat(self, other: Index) -> Categorical:
        # [FUNCTION DOC]
        # HELPER: used by equality and append logic to coerce another index
        # to a Categorical compatible with this one. If other is categorical,
        # its categories must match this index's up to permutation, else a
        # type error is raised; it is returned as a Categorical. A
        # MultiIndex raises a type error. Otherwise every non-missing value
        # of other must be among this index's categories (else type error),
        # and the values are converted to a Categorical with this dtype;
        # a type error is also raised if any value would not be preserved.
        # Returns the Categorical. No mutation.
        """
        *this is an internal non-public method*

        provide a comparison between the dtype of self and other (coercing if
        needed)

        Parameters
        ----------
        other : Index

        Returns
        -------
        Categorical

        Raises
        ------
        TypeError if the dtypes are not compatible
        """
        if isinstance(other.dtype, CategoricalDtype):
            cat = extract_array(other)
            cat = cast(Categorical, cat)
            if not cat._categories_match_up_to_permutation(self._values):
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        elif other._is_multi:
            # preempt raising NotImplementedError in isna call
            raise TypeError("MultiIndex is not dtype-compatible with CategoricalIndex")
        else:
            values = other

            codes = self.categories.get_indexer(values)
            if ((codes == -1) & ~values.isna()).any():
                # GH#37667 see test_equals_non_category
                raise TypeError(
                    "categories must match existing categories when appending"
                )
            cat = Categorical(other, dtype=self.dtype)
            other = CategoricalIndex(cat)
            if not other.isin(values).all():
                raise TypeError(
                    "cannot append a non-category item to a CategoricalIndex"
                )
            cat = other._values

        return cat

    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Checks equality with other. Returns true when other is the same
        # object; false when other is not an Index; otherwise coerces other
        # to a compatible Categorical and compares the underlying data
        # (element order, orderedness, and categories all matter; category
        # order matters only for ordered data). Incompatible dtypes (type or
        # value errors from coercion) yield false rather than raising.
        """
        Determine if two CategoricalIndex objects contain the same elements.

        The order and orderedness of elements matters. The categories matter,
        but the order of the categories matters only when ``ordered=True``.

        Parameters
        ----------
        other : object
            The CategoricalIndex object to compare with.

        Returns
        -------
        bool
            ``True`` if two :class:`pandas.CategoricalIndex` objects have equal
            elements, ``False`` otherwise.

        See Also
        --------
        Categorical.equals : Returns True if categorical arrays are equal.

        Examples
        --------
        >>> ci = pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
        >>> ci2 = pd.CategoricalIndex(pd.Categorical(["a", "b", "c", "a", "b", "c"]))
        >>> ci.equals(ci2)
        True

        The order of elements matters.

        >>> ci3 = pd.CategoricalIndex(["c", "b", "a", "a", "b", "c"])
        >>> ci.equals(ci3)
        False

        The orderedness also matters.

        >>> ci4 = ci.as_ordered()
        >>> ci.equals(ci4)
        False

        The categories matter, but the order of the categories matters only when
        ``ordered=True``.

        >>> ci5 = ci.set_categories(["a", "b", "c", "d"])
        >>> ci.equals(ci5)
        False

        >>> ci6 = ci.set_categories(["b", "c", "a"])
        >>> ci.equals(ci6)
        True
        >>> ci_ordered = pd.CategoricalIndex(
        ...     ["a", "b", "c", "a", "b", "c"], ordered=True
        ... )
        >>> ci2_ordered = ci_ordered.set_categories(["b", "c", "a"])
        >>> ci_ordered.equals(ci2_ordered)
        False
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        try:
            other = self._is_dtype_compat(other)
        except (TypeError, ValueError):
            return False

        return self._data.equals(other)

    # --------------------------------------------------------------------
    # Rendering Methods

    @property
    def _formatter_func(self):
        # [FUNCTION DOC]
        # HELPER: property for repr formatting. Returns the element formatter
        # of the categories index, used to render each value.
        return self.categories._formatter_func

    def _format_attrs(self):
        # [FUNCTION DOC]
        # HELPER: for the repr. Returns a list of (name, formatted value)
        # pairs: first the categories (comma-joined truncated repr, in
        # brackets), then the ordered flag, followed by the attributes added
        # by the base class implementation (e.g. dtype, length, name).
        """
        Return a list of tuples of the (attr,formatted_value)
        """
        attrs: list[tuple[str, str | int | bool | None]]

        attrs = [
            (
                "categories",
                f"[{', '.join(self._data._repr_categories())}]",
            ),
            ("ordered", self.ordered),
        ]
        extra = super()._format_attrs()
        return attrs + extra

    # --------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        # [FUNCTION DOC]
        # Property; always returns the constant string "categorical",
        # regardless of the contents.
        return "categorical"

    def __contains__(self, key: Any) -> bool:
        # [FUNCTION DOC]
        # Membership test. Parameter key is any label. Returns a bool. If key
        # is a missing-value marker valid for the categories dtype, returns
        # whether the index contains any missing values. Otherwise delegates
        # to the categorical containment helper, using the categories index
        # itself as lookup container when it is a range index, else this
        # index's engine. Unhashable keys raise a type error (as documented).
        # KNOWN ISSUE: the docstring example uses a plain integer Index
        # rather than a CategoricalIndex, so it does not demonstrate this
        # method's categorical behavior.
        """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        Raises
        ------
        TypeError
            If the key is not hashable.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')

        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """
        # if key is a NaN, check if any NaN is in self.
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return self.hasnans
        if self.categories._typ == "rangeindex":
            container: Index | libindex.IndexEngine | libindex.ExtensionEngine = (
                self.categories
            )
        else:
            container = self._engine
        return contains(self, key, container=container)

    def reindex(
        self, target, method=None, level=None, limit: int | None = None, tolerance=None
    ) -> tuple[Index, npt.NDArray[np.intp] | None]:
        # [FUNCTION DOC]
        # Conform the index to target. Parameters: target is the new labels
        # (array-like or Index); method, level and limit must all be none;
        # tolerance is accepted but ignored. Raises a not-implemented error
        # if method, level or limit is given. Returns a pair of the new index
        # and an optional integer indexer into the original (none if the
        # index already matches), computed by the base class implementation
        # called with target only.
        # KNOWN ISSUE: the tolerance parameter is silently ignored and is not
        # rejected like the other unsupported arguments.
        """
        Create index with target's values (move/add/delete values as necessary)

        Returns
        -------
        new_index : pd.Index
            Resulting index
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index

        """
        if method is not None:
            raise NotImplementedError(
                "argument method is not implemented for CategoricalIndex.reindex"
            )
        if level is not None:
            raise NotImplementedError(
                "argument level is not implemented for CategoricalIndex.reindex"
            )
        if limit is not None:
            raise NotImplementedError(
                "argument limit is not implemented for CategoricalIndex.reindex"
            )
        return super().reindex(target)

    # --------------------------------------------------------------------
    # Indexing Methods

    def _maybe_cast_indexer(self, key) -> int:
        # [FUNCTION DOC]
        # HELPER: for scalar label lookup. Converts key into its integer
        # code via the categorical array's unboxing (which handles partial
        # indexing on interval categories). Returns the code as an int. If
        # the key is not a category but is a missing marker valid for the
        # categories dtype, returns minus one; otherwise re-raises the key
        # error.
        # GH#41933: we have to do this instead of self._data._validate_scalar
        #  because this will correctly get partial-indexing on Interval categories
        try:
            return self._data._unbox_scalar(key)
        except KeyError:
            if is_valid_na_for_dtype(key, self.categories.dtype):
                return -1
            raise

    def _maybe_cast_listlike_indexer(self, values) -> CategoricalIndex:
        # [FUNCTION DOC]
        # HELPER: for list-like lookup. Converts values into a
        # CategoricalIndex expressed in this index's categories. A
        # categorical input (index or array) is re-encoded with this index's
        # categories; any other input is mapped to category positions through
        # the categories index (missing for absent labels) and cast to the
        # codes dtype. Returns a new CategoricalIndex; no mutation.
        if isinstance(values, CategoricalIndex):
            values = values._data
        if isinstance(values, Categorical):
            # Indexing on codes is more efficient if categories are the same,
            #  so we can apply some optimizations based on the degree of
            #  dtype-matching.
            cat = self._data._encode_with_my_categories(values)
            codes = cat._codes
        else:
            codes = self.categories.get_indexer(values)
            codes = codes.astype(self.codes.dtype, copy=False)
            cat = self._data._from_backing_data(codes)
        return type(self)._simple_new(cat)

    # --------------------------------------------------------------------

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        # [FUNCTION DOC]
        # HELPER: Parameter dtype is a dtype object. Returns whether values
        # of that dtype are comparable with this index, delegating to the
        # categories index.
        return self.categories._is_comparable_dtype(dtype)

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        # [FUNCTION DOC]
        # Map the categories through mapper (a function, dict or Series).
        # na_action is either ignore (missing values are not passed to the
        # mapper) or none. Delegates to the categorical array's map, then
        # wraps the result in an Index with the same name (no copy). Returns
        # a CategoricalIndex when the mapping is one-to-one, otherwise a
        # plain Index (determined by the array result type).
        # KNOWN ISSUE: the docstring says na_action defaults to ignore, but
        # the signature default is none.
        """
        Map values using input an input mapping or function.

        Maps the values (their categories, not the codes) of the index to new
        categories. If the mapping correspondence is one-to-one the result is a
        :class:`~pandas.CategoricalIndex` which has the same order property as
        the original, otherwise an :class:`~pandas.Index` is returned.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default 'ignore'
            If 'ignore', propagate NaN values, without passing them to
            the mapping correspondence.

        Returns
        -------
        pandas.CategoricalIndex or pandas.Index
            Mapped index.

        See Also
        --------
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> idx = pd.CategoricalIndex(["a", "b", "c"])
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                          ordered=False, dtype='category')
        >>> idx.map(lambda x: x.upper())
        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'],
                         ordered=False, dtype='category')
        >>> idx.map({"a": "first", "b": "second", "c": "third"})
        CategoricalIndex(['first', 'second', 'third'], categories=['first',
                         'second', 'third'], ordered=False, dtype='category')

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> idx = pd.CategoricalIndex(["a", "b", "c"], ordered=True)
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> idx.map({"a": 3, "b": 2, "c": 1})
        CategoricalIndex([3, 2, 1], categories=[3, 2, 1], ordered=True,
                         dtype='category')

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> idx.map({"a": "first", "b": "second", "c": "first"})
        Index(['first', 'second', 'first'], dtype='str')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> idx.map({"a": "first", "b": "second"})
        Index(['first', 'second', nan], dtype='str')
        """
        mapped = self._values.map(mapper, na_action=na_action)
        return Index(mapped, name=self.name, copy=False)
