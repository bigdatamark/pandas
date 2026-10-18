"""
Define extension dtypes.
"""

# [MODULE DOC]
# Concrete pandas extension dtypes. Public classes (exposed at the top level):
# CategoricalDtype, DatetimeTZDtype, PeriodDtype, IntervalDtype, SparseDtype and
# ArrowDtype. Internal classes: PandasExtensionDtype (shared base for dtypes that mimic
# numpy dtype attributes), CategoricalDtypeType (scalar-type placeholder for
# categoricals), NumpyEADtype (wraps a numpy dtype as an extension dtype) and
# BaseMaskedDtype (base for nullable integer, floating and boolean dtypes defined in the
# arrays package).
# Relationships: all derive from the abstract extension dtype and storage-extension
# dtype in the base module and (except the internal ones) register themselves with the
# dtype registry through the registration decorator, which makes string aliases
# resolvable. They pair with array classes imported lazily from the arrays package.
# Depend on the generic ABC classes and inference predicates, and use compiled
# time-zone, offset and period helpers. The common, cast and missing modules of this
# package import from here, so many reverse imports are performed inside function bodies
# to avoid cycles.
# Import-time side effects: registers six dtype classes in the global registry;
# conditionally imports pyarrow when it is available; defines an alias for the builtin
# string type used in annotations because several classes define a class attribute named
# str.


from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    cast,
)
import warnings
import zoneinfo

import numpy as np

from pandas._config.config import get_option

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Period,
    Timedelta,
    Timestamp,
    timezones,
    to_offset,
    tz_compare,
)
from pandas._libs.tslibs.dtypes import (
    PeriodDtypeBase,
    abbrev_to_npy_unit,
)
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import (
    HAS_PYARROW,
    PYARROW_MIN_VERSION,
)
from pandas.errors import PerformanceWarning
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.inference import (
    is_bool,
    is_list_like,
)

if HAS_PYARROW:
    import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from datetime import tzinfo

    import pyarrow as pa  # noqa: TC004

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Ordered,
        Scalar,
        TimeUnit,
        npt,
        type_t,
    )

    from pandas import (
        Categorical,
        CategoricalIndex,
        DatetimeIndex,
        Index,
        IntervalIndex,
        PeriodIndex,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        DatetimeArray,
        IntervalArray,
        NumpyExtensionArray,
        PeriodArray,
        SparseArray,
    )
    from pandas.core.arrays.arrow import ArrowExtensionArray

str_type = str


class PandasExtensionDtype(ExtensionDtype):
    # [CLASS DOC]
    # Base class for extension dtypes that duck-type numpy dtypes ("not a real numpy
    # dtype"). Provides default numpy-like class attributes (subdtype, number code,
    # shape, itemsize, base, builtin and native flags) and a per-class cache dictionary
    # for flyweight instances. Subclasses must implement hashing (the base raises a
    # not-implemented error). Pickling stores only the attributes named in the metadata
    # tuple, never the cache.
    """
    An np.dtype duck-typed class, suitable for holding a custom dtype.

    THIS IS NOT A REAL NUMPY DTYPE
    """

    type: Any
    kind: Any
    # The Any type annotations above are here only because mypy seems to have a
    # problem dealing with multiple inheritance from PandasExtensionDtype
    # and ExtensionDtype's @properties in the subclasses below. The kind and
    # type variables in those subclasses are explicitly typed below.
    subdtype: DtypeObj | None = None
    str: str_type
    num = 100
    shape: tuple[int, ...] = ()
    itemsize = 8
    base: DtypeObj | None = None
    isbuiltin = 0
    isnative = 0
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}

    def __repr__(self) -> str_type:
        # [FUNCTION DOC]
        # Returns the same text as the string conversion.
        """
        Return a string representation for a particular object.
        """
        return str(self)

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Always raises a not-implemented error; subclasses must supply their own hash.
        raise NotImplementedError("sub-classes should implement an __hash__ method")

    def __getstate__(self) -> dict[str_type, Any]:
        # [FUNCTION DOC]
        # Pickle support. Returns a dictionary mapping each attribute named in the
        # metadata tuple to its current value (or none if absent); the instance cache is
        # not included.
        # pickle support; we don't want to pickle the cache
        return {k: getattr(self, k, None) for k in self._metadata}

    @classmethod
    def reset_cache(cls) -> None:
        # [FUNCTION DOC]
        # Class method. Replaces the class-level cache dictionary with a new empty one
        # (side effect: affects only the class it is called on).
        """clear the cache"""
        cls._cache_dtypes = {}


class CategoricalDtypeType(type):
    # [CLASS DOC]
    # Empty subclass of the type builtin used as the scalar "type" attribute of
    # categorical dtypes; it exists so the scalar type is distinguishable and subclass
    # checks can be customised. No state or methods.
    """
    the type of CategoricalDtype, this metaclass determines subclass ability
    """


@register_extension_dtype
@set_module("pandas")
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    # [CLASS DOC]
    # Extension dtype for categorical data: an ordered or unordered fixed set of unique,
    # non-null categories held in an Index. Registered under the name "category". State:
    # private categories (an Index, or none for an uninitialized dtype) and ordered
    # (boolean, or none meaning "unspecified, preserve existing"). Identity metadata is
    # the pair of categories and ordered. Class attributes mirror numpy-like fields
    # (object kind, object base) and declare no 2D support. Paired with the Categorical
    # array and CategoricalIndex. Key methods: validators for ordered and categories,
    # construction helpers from strings, other dtypes or values, update of one dtype by
    # another, equality/hash rules, and common-dtype resolution.
    """
    Type for categorical data with the categories and orderedness.

    It is a dtype representation for categorical data, which allows users to define
    a fixed set of values and optionally impose an ordering. This is particularly
    useful for handling categorical variables efficiently, as it can significantly
    reduce memory usage compared to using object dtypes.

    Parameters
    ----------
    categories : sequence, optional
        Must be unique, and must not contain any nulls.
        The categories are stored in an Index,
        and if an index is provided the dtype of that index will be used.
    ordered : bool or None, default False
        Whether or not this categorical is treated as an ordered categorical.
        None can be used to maintain the ordered value of existing categoricals when
        used in operations that combine categoricals, e.g. astype, and will resolve to
        False if there is no existing ordered to maintain.

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    None

    See Also
    --------
    Categorical : Represent a categorical variable in classic R / S-plus fashion.

    Notes
    -----
    This class is useful for specifying the type of a ``Categorical``
    independent of the values. See :ref:`categorical.categoricaldtype`
    for more.

    Examples
    --------
    >>> t = pd.CategoricalDtype(categories=["b", "a"], ordered=True)
    >>> pd.Series(["a", "b", "a", None], dtype=t)
    0      a
    1      b
    2      a
    3    NaN
    dtype: category
    Categories (2, str): ['b' < 'a']

    An empty CategoricalDtype with a specific dtype can be created
    by providing an empty index. As follows,

    >>> pd.CategoricalDtype(pd.DatetimeIndex([])).categories.dtype
    dtype('<M8[s]')
    """

    # TODO: Document public vs. private API
    name = "category"
    type: type[CategoricalDtypeType] = CategoricalDtypeType
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    _metadata = ("categories", "ordered")
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, categories=None, ordered: Ordered = False) -> None:
        # [FUNCTION DOC]
        # Parameters: categories (default none), a sequence of unique non-null values
        # stored as an Index (an Index keeps its own dtype); ordered (default false),
        # true, false or none. Validates both through the shared finalizing routine (so
        # invalid input raises type errors for non-boolean ordered, type error for
        # non-list-like categories, value errors for nulls or duplicates).
        self._finalize(categories, ordered, fastpath=False)

    @classmethod
    def _from_fastpath(
        cls, categories=None, ordered: bool | None = None
    ) -> CategoricalDtype:
        # [FUNCTION DOC]
        # HELPER: class method creating an instance without running the constructor,
        # then finalizing with the fast path that skips the null/uniqueness/list-like
        # checks on categories. Parameters: categories (default none), ordered (default
        # none). Returns the new dtype. Callers must guarantee valid categories.
        self = cls.__new__(cls)
        self._finalize(categories, ordered, fastpath=True)
        return self

    @classmethod
    def _from_categorical_dtype(
        cls, dtype: CategoricalDtype, categories=None, ordered: Ordered | None = None
    ) -> CategoricalDtype:
        # [FUNCTION DOC]
        # HELPER: class method. Parameters: dtype, an existing CategoricalDtype;
        # categories and ordered (both optional overrides). If both overrides are none
        # returns the given dtype itself; otherwise builds a new dtype using the
        # override when given, else the existing value.
        if categories is ordered is None:
            return dtype
        if categories is None:
            categories = dtype.categories
        if ordered is None:
            ordered = dtype.ordered
        return cls(categories, ordered)

    @classmethod
    def _from_values_or_dtype(
        cls,
        values=None,
        categories=None,
        ordered: bool | None = None,
        dtype: Dtype | None = None,
    ) -> CategoricalDtype:
        # [FUNCTION DOC]
        # HELPER: class method building a dtype from the inputs of the Categorical
        # constructor, without factorizing values. Parameters: values (list-like or
        # categorical, optional), categories, ordered, dtype (a CategoricalDtype or the
        # string "category", optional). Priority: (1) a dtype argument: the string
        # "category" builds a dtype from categories and ordered (keeping the values'
        # orderedness if ordered is none and values are categorical), any other string
        # raises a value error, a non-categorical dtype object raises a value error, and
        # passing categories or ordered together with a dtype object raises a value
        # error; a CategoricalDtype is returned as is. (2) Without dtype, categorical
        # values contribute their dtype, overridden by non-none categories and ordered.
        # (3) Otherwise a new dtype from categories and ordered, which may have both
        # none. Returns the dtype.
        # KNOWN ISSUE: the error message for an unsupported dtype object reads "Cannot
        # not construct" (a duplicated negation typo).
        """
        Construct dtype from the input parameters used in :class:`Categorical`.

        This constructor method specifically does not do the factorization
        step, if that is needed to find the categories. This constructor may
        therefore return ``CategoricalDtype(categories=None, ordered=None)``,
        which may not be useful. Additional steps may therefore have to be
        taken to create the final dtype.

        The return dtype is specified from the inputs in this prioritized
        order:
        1. if dtype is a CategoricalDtype, return dtype
        2. if dtype is the string 'category', create a CategoricalDtype from
           the supplied categories and ordered parameters, and return that.
        3. if values is a categorical, use value.dtype, but override it with
           categories and ordered if either/both of those are not None.
        4. if dtype is None and values is not a categorical, construct the
           dtype from categories and ordered, even if either of those is None.

        Parameters
        ----------
        values : list-like, optional
            The list-like must be 1-dimensional.
        categories : list-like, optional
            Categories for the CategoricalDtype.
        ordered : bool, optional
            Designating if the categories are ordered.
        dtype : CategoricalDtype or the string "category", optional
            If ``CategoricalDtype``, cannot be used together with
            `categories` or `ordered`.

        Returns
        -------
        CategoricalDtype

        Examples
        --------
        >>> pd.CategoricalDtype._from_values_or_dtype()
        CategoricalDtype(categories=None, ordered=None, categories_dtype=None)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     categories=["a", "b"], ordered=True
        ... )
        CategoricalDtype(categories=['a', 'b'], ordered=True, categories_dtype=str)
        >>> dtype1 = pd.CategoricalDtype(["a", "b"], ordered=True)
        >>> dtype2 = pd.CategoricalDtype(["x", "y"], ordered=False)
        >>> c = pd.Categorical([0, 1], dtype=dtype1)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     c, ["x", "y"], ordered=True, dtype=dtype2
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cannot specify `categories` or `ordered` together with
        `dtype`.

        The supplied dtype takes precedence over values' dtype:

        >>> pd.CategoricalDtype._from_values_or_dtype(c, dtype=dtype2)
        CategoricalDtype(categories=['x', 'y'], ordered=False, categories_dtype=str)
        """

        if dtype is not None:
            # The dtype argument takes precedence over values.dtype (if any)
            if isinstance(dtype, str):
                if dtype == "category":
                    if ordered is None and cls.is_dtype(values):
                        # GH#49309 preserve orderedness
                        ordered = values.dtype.ordered

                    dtype = CategoricalDtype(categories, ordered)
                else:
                    raise ValueError(f"Unknown dtype {dtype!r}")
            elif categories is not None or ordered is not None:
                raise ValueError(
                    "Cannot specify `categories` or `ordered` together with `dtype`."
                )
            elif not isinstance(dtype, CategoricalDtype):
                raise ValueError(f"Cannot not construct CategoricalDtype from {dtype}")
        elif cls.is_dtype(values):
            # If no "dtype" was passed, use the one from "values", but honor
            # the "ordered" and "categories" arguments
            dtype = values.dtype._from_categorical_dtype(
                values.dtype, categories, ordered
            )
        else:
            # If dtype=None and values is not categorical, create a new dtype.
            # Note: This could potentially have categories=None and
            # ordered=None.
            dtype = CategoricalDtype(categories, ordered)

        return cast(CategoricalDtype, dtype)

    @classmethod
    def construct_from_string(cls, string: str_type) -> CategoricalDtype:
        # [FUNCTION DOC]
        # Class method. Parameter: string. Raises a type error if it is not a string or
        # is not exactly the name "category". Returns a dtype with no categories and
        # ordered set to none (so that casting to the string alias does not override
        # existing orderedness).
        """
        Construct a CategoricalDtype from a string.

        Parameters
        ----------
        string : str
            Must be the string "category" in order to be successfully constructed.

        Returns
        -------
        CategoricalDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a CategoricalDtype cannot be constructed from the input.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string != cls.name:
            raise TypeError(f"Cannot construct a 'CategoricalDtype' from '{string}'")

        # need ordered=None to ensure that operations specifying dtype="category" don't
        # override the ordered value for existing categoricals
        return cls(ordered=None)

    def _finalize(self, categories, ordered: Ordered, fastpath: bool = False) -> None:
        # [FUNCTION DOC]
        # HELPER: shared initializer. Parameters: categories, ordered, fastpath (default
        # false). Validates ordered when not none, validates and converts categories to
        # an Index when not none (fastpath skips checks), then stores both on the
        # instance.
        if ordered is not None:
            self.validate_ordered(ordered)

        if categories is not None:
            categories = self.validate_categories(categories, fastpath=fastpath)

        self._categories = categories
        self._ordered = ordered

    def __setstate__(self, state: MutableMapping[str_type, Any]) -> None:
        # [FUNCTION DOC]
        # Unpickling support. Parameter: state, a mapping. Restores the private
        # categories (default none) and ordered (default false) from the keys of the
        # same names, removing them from the mapping.
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._categories = state.pop("categories", None)
        self._ordered = state.pop("ordered", False)

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Returns minus one for an uncategorized ordered dtype and minus two for an
        # uncategorized unordered one (negative space, since category hashes are
        # unsigned); otherwise the integer form of the cached category hash, which
        # includes the orderedness.
        # _hash_categories returns a uint64, so use the negative
        # space for when we have unknown categories to avoid a conflict
        if self.categories is None:
            if self.ordered:
                return -1
            else:
                return -2
        # We *do* want to include the real self.ordered here
        return int(self._hash_categories)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Rules (from the code): a string equals the dtype if it is the name "category";
        # identity is equal; objects without both orderedness and categories attributes
        # are unequal; if either side has no categories, equal only when both lack
        # categories; if either is ordered, equal only when both are ordered with equal
        # categories in the same order; if neither is ordered (none and false are the
        # same), equal when category dtypes match, lengths match and the categories are
        # the same set (identical sequence fast path; position lookup for non-object
        # dtypes; set comparison for object dtypes so integer and float values stay
        # distinct).
        # KNOWN ISSUE: the docstring's third rule says a dtype is equal to any dtype
        # with no categories regardless of orderedness, but the code only treats a dtype
        # without categories as equal to another dtype without categories.
        """
        Rules for CDT equality:
        1) Any CDT is equal to the string 'category'
        2) Any CDT is equal to itself
        3) Any CDT is equal to a CDT with categories=None regardless of ordered
        4) A CDT with ordered=True is only equal to another CDT with
           ordered=True and identical categories in the same order
        5) A CDT with ordered={False, None} is only equal to another CDT with
           ordered={False, None} and identical categories, but same order is
           not required. There is no distinction between False/None.
        6) Any other comparison returns False
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not (hasattr(other, "ordered") and hasattr(other, "categories")):
            return False
        elif self.categories is None or other.categories is None:
            # For non-fully-initialized dtypes, these are only equal to
            #  - the string "category" (handled above)
            #  - other CategoricalDtype with categories=None
            return self.categories is other.categories
        elif self.ordered or other.ordered:
            # At least one has ordered=True; equal if both have ordered=True
            # and the same values for categories in the same order.
            return (self.ordered == other.ordered) and self.categories.equals(
                other.categories
            )
        else:
            # Neither has ordered=True; equal if both have the same categories,
            # but same order is not necessary.  There is no distinction between
            # ordered=False and ordered=None: CDT(., False) and CDT(., None)
            # will be equal if they have the same categories.
            left = self.categories
            right = other.categories

            # GH#36280 the ordering of checks here is for performance
            if not left.dtype == right.dtype:
                return False

            if len(left) != len(right):
                return False

            if self.categories.equals(other.categories):
                # Check and see if they happen to be identical categories
                return True

            if left.dtype != object:
                # Faster than calculating hash
                indexer = left.get_indexer(right)
                # Because left and right have the same length and are unique,
                #  `indexer` not having any -1s implies that there is a
                #  bijection between `left` and `right`.
                return bool((indexer != -1).all())

            # With object-dtype we need a comparison that identifies
            #  e.g. int(2) as distinct from float(2)
            return set(left) == set(right)

    def __repr__(self) -> str_type:
        # [FUNCTION DOC]
        # Returns a string naming the class with the formatted categories (a range index
        # shows its range; trailing separators stripped; the word None when
        # uninitialized), the orderedness and the categories' dtype.
        if self.categories is None:
            data = "None"
            dtype = "None"
        else:
            data = self.categories._format_data(name=type(self).__name__)
            if isinstance(self.categories, ABCRangeIndex):
                data = str(self.categories._range)
            data = data.rstrip(", ")
            dtype = self.categories.dtype

        return (
            f"CategoricalDtype(categories={data}, ordered={self.ordered}, "
            f"categories_dtype={dtype})"
        )

    @cache_readonly
    def _hash_categories(self) -> int:
        # [FUNCTION DOC]
        # HELPER: cached. Computes the integer hash used by the hash method, from the
        # categories and orderedness. Tuple categories are hashed as tuples; object
        # categories of mixed types use the builtin hash of the category tuple combined
        # with orderedness and return early; timezone-aware categories are viewed as
        # nanosecond datetimes; the values are hashed with the array hash, an ordered
        # dtype combines each element hash with its position, and the hashes are
        # combined by xor. Requires initialized categories.
        from pandas.core.util.hashing import (
            combine_hash_arrays,
            hash_array,
            hash_tuples,
        )

        categories = self.categories
        ordered = self.ordered

        if len(categories) and isinstance(categories[0], tuple):
            # assumes if any individual category is a tuple, then all our. ATM
            # I don't really want to support just some of the categories being
            # tuples.
            cat_list = list(categories)  # breaks if an np.array of categories
            cat_array = hash_tuples(cat_list)
        else:
            if categories.dtype == "O" and len({type(x) for x in categories}) != 1:
                # TODO: hash_array doesn't handle mixed types. It casts
                # everything to a str first, which means we treat
                # {'1', '2'} the same as {'1', 2}
                # find a better solution
                hashed = hash((tuple(categories), ordered))
                return hashed

            if DatetimeTZDtype.is_dtype(categories.dtype):
                # Avoid future warning.
                categories = categories.view("datetime64[ns]")

            cat_array = hash_array(np.asarray(categories), categorize=False)
        if ordered:
            cat_array = np.vstack(
                [cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)]
            )
        else:
            cat_array = cat_array.reshape(1, len(cat_array))
        combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(combined_hashed)

    def construct_array_type(self) -> type_t[Categorical]:
        # [FUNCTION DOC]
        # Returns the Categorical array class (imported lazily).
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas import Categorical

        return Categorical

    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        # [FUNCTION DOC]
        # Static method. Parameter: ordered. Raises a type error unless it is a boolean
        # (as judged by the boolean predicate). Returns nothing.
        """
        Validates that we have a valid ordered parameter. If
        it is not a boolean, a TypeError will be raised.

        Parameters
        ----------
        ordered : object
            The parameter to be verified.

        Raises
        ------
        TypeError
            If 'ordered' is not a boolean.
        """
        if not is_bool(ordered):
            raise TypeError("'ordered' must either be 'True' or 'False'")

    @staticmethod
    def validate_categories(categories, fastpath: bool = False) -> Index:
        # [FUNCTION DOC]
        # Static method. Parameters: categories, array-like; fastpath (default false).
        # Without the fast path raises a type error if not list-like. Converts non-Index
        # input to an Index with type inference and without tuples becoming a
        # multi-level index. Without the fast path also raises a value error if the
        # categories contain nulls or duplicates. A categorical index is replaced by its
        # own categories. Returns an Index.
        # KNOWN ISSUE: the docstring says the fast path only skips the null and
        # uniqueness checks, but it also skips the list-like check.
        """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
        fastpath : bool
            Whether to skip nan and uniqueness checks

        Returns
        -------
        categories : Index
        """
        from pandas.core.indexes.base import Index

        if not fastpath and not is_list_like(categories):
            raise TypeError(
                f"Parameter 'categories' must be list-like, was {categories!r}"
            )
        if not isinstance(categories, ABCIndex):
            categories = Index._with_infer(categories, tupleize_cols=False)

        if not fastpath:
            if categories.hasnans:
                raise ValueError("Categorical categories cannot be null")

            if not categories.is_unique:
                raise ValueError("Categorical categories must be unique")

        if isinstance(categories, ABCCategoricalIndex):
            categories = categories.categories

        return categories

    def update_dtype(self, dtype: str_type | CategoricalDtype) -> CategoricalDtype:
        # [FUNCTION DOC]
        # Parameter: dtype, a CategoricalDtype or the string "category". The string
        # returns self unchanged; anything else that is not a categorical dtype raises a
        # value error. If the given dtype has both categories and orderedness set it is
        # returned as is; otherwise a new dtype is made using the given
        # categories/orderedness where present and self's where the given is none.
        """
        Returns a CategoricalDtype with categories and ordered taken from dtype
        if specified, otherwise falling back to self if unspecified

        Parameters
        ----------
        dtype : CategoricalDtype

        Returns
        -------
        new_dtype : CategoricalDtype
        """
        if isinstance(dtype, str) and dtype == "category":
            # dtype='category' should not change anything
            return self
        elif not self.is_dtype(dtype):
            raise ValueError(
                f"a CategoricalDtype must be passed to perform an update, got {dtype!r}"
            )
        else:
            # from here on, dtype is a CategoricalDtype
            dtype = cast(CategoricalDtype, dtype)

        # update categories/ordered unless they've been explicitly passed as None
        if (
            isinstance(dtype, CategoricalDtype)
            and dtype.categories is not None
            and dtype.ordered is not None
        ):
            # Avoid re-validation in CategoricalDtype constructor
            return dtype
        new_categories = (
            dtype.categories if dtype.categories is not None else self.categories
        )
        new_ordered = dtype.ordered if dtype.ordered is not None else self.ordered

        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self) -> Index:
        # [FUNCTION DOC]
        # Property returning the stored Index of categories (none when uninitialized).
        """
        An ``Index`` containing the unique categories allowed.

        See Also
        --------
        ordered : Whether the categories have an ordered relationship.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=["a", "b"], ordered=True)
        >>> cat_type.categories
        Index(['a', 'b'], dtype='str')
        """
        return self._categories

    @property
    def ordered(self) -> Ordered:
        # [FUNCTION DOC]
        # Property returning the orderedness flag: true, false or none.
        """
        Whether the categories have an ordered relationship.

        See Also
        --------
        categories : An Index containing the unique categories allowed.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=["a", "b"], ordered=True)
        >>> cat_type.ordered
        True

        >>> cat_type = pd.CategoricalDtype(categories=["a", "b"], ordered=False)
        >>> cat_type.ordered
        False
        """
        return self._ordered

    @property
    def _is_boolean(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property. True when the boolean predicate judges the categories Index
        # to be boolean (so a categorical of booleans is boolean). Imports the predicate
        # lazily.
        from pandas.core.dtypes.common import is_bool_dtype

        return is_bool_dtype(self.categories)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: common-type hook. If all dtypes are categorical and equal to the
        # first, returns the first. If all are uninitialized categoricals returns self;
        # if only some are, returns none. Otherwise sparse dtypes are replaced by their
        # subtype and categorical ones by their categories' dtype, and the generic
        # common-type finder decides. Returns a dtype or none.
        # check if we have all categorical dtype with identical categories
        if all(isinstance(x, CategoricalDtype) for x in dtypes):
            first = dtypes[0]
            if all(first == other for other in dtypes[1:]):
                return first

        # special case non-initialized categorical
        # TODO we should figure out the expected return value in general
        non_init_cats = [
            isinstance(x, CategoricalDtype) and x.categories is None for x in dtypes
        ]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None

        # categorical is aware of Sparse -> extract sparse subdtypes
        subtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        # extract the categories' dtype
        non_cat_dtypes = [
            x.categories.dtype if isinstance(x, CategoricalDtype) else x
            for x in subtypes
        ]
        # TODO should categorical always give an answer?
        from pandas.core.dtypes.cast import find_common_type

        return find_common_type(non_cat_dtypes)

    @cache_readonly
    def index_class(self) -> type_t[CategoricalIndex]:
        # [FUNCTION DOC]
        # Cached. Returns the categorical index class used for indexes of this dtype
        # (imported lazily).
        from pandas import CategoricalIndex

        return CategoricalIndex


@register_extension_dtype
@set_module("pandas")
class DatetimeTZDtype(PandasExtensionDtype):
    # [CLASS DOC]
    # Extension dtype for timezone-aware datetime data (a duck-typed numpy-like dtype).
    # Parameters (constructor): unit and tz. State: private unit string (one of s, ms,
    # us, ns) and tz object. Identity metadata: unit and tz. Scalar type is Timestamp,
    # kind code is datetime, NaT is the missing value, and it supports two-dimensional
    # arrays and fast transpose. Pairs with the datetime array and datetime index.
    # Provides parsing of its string form, equality by unit and timezone comparison,
    # pyarrow conversion and common-type resolution.
    """
    An ExtensionDtype for timezone-aware datetime data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    unit : str, default "ns"
        The precision of the datetime data. Valid options are
        ``"s"``, ``"ms"``, ``"us"``, ``"ns"``.
    tz : str, int, or datetime.tzinfo
        The timezone.

    Attributes
    ----------
    unit
    tz

    Methods
    -------
    None

    Raises
    ------
    ZoneInfoNotFoundError
        When the requested timezone cannot be found.

    See Also
    --------
    numpy.datetime64 : Numpy data type for datetime.
    datetime.datetime : Python datetime object.

    Examples
    --------
    >>> from zoneinfo import ZoneInfo
    >>> pd.DatetimeTZDtype(tz=ZoneInfo("UTC"))
    datetime64[ns, UTC]

    >>> pd.DatetimeTZDtype(tz=ZoneInfo("Europe/Paris"))
    datetime64[ns, Europe/Paris]
    """

    type: type[Timestamp] = Timestamp
    kind: str_type = "M"
    num = 101
    _metadata = ("unit", "tz")
    _match = re.compile(r"(datetime64|M8)\[(?P<unit>.+), (?P<tz>.+)\]")
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = True
    _can_fast_transpose = True

    @property
    def na_value(self) -> NaTType:
        # [FUNCTION DOC]
        # Property returning the NaT singleton.
        return NaT

    @cache_readonly
    def base(self) -> DtypeObj:  # type: ignore[override]
        # [FUNCTION DOC]
        # Cached. Returns the numpy datetime64 dtype with the same unit (the tz-naive
        # storage dtype).
        return np.dtype(f"M8[{self.unit}]")

    # error: Signature of "str" incompatible with supertype "PandasExtensionDtype"
    @cache_readonly
    def str(self) -> str:  # type: ignore[override]
        # [FUNCTION DOC]
        # Cached. Returns the numpy-style string form: a bar, the datetime64 short code
        # and the unit in brackets.
        return f"|M8[{self.unit}]"

    def __init__(self, unit: TimeUnit | DatetimeTZDtype = "ns", tz=None) -> None:
        # [FUNCTION DOC]
        # Parameters: unit (default nanoseconds), either a unit string or another
        # DatetimeTZDtype (whose unit and tz are copied); tz, a time zone name, offset
        # or tzinfo object. A non-nanosecond string unit with no tz is treated as an
        # old-style alias (e.g. a full dtype string): it is parsed and then a value
        # error is always raised telling the user to use the string constructor instead.
        # Units outside seconds, milliseconds, microseconds, nanoseconds raise a value
        # error. A truthy tz is resolved and standardized; a falsy but not none tz (such
        # as an empty string) raises the zone-not-found error; a missing tz raises a
        # type error.
        # KNOWN ISSUE: the alias branch always ends in a value error even when parsing
        # succeeds, so the parsed unit assigned just before is never used afterwards (only the zone feeds the message).
        if isinstance(unit, DatetimeTZDtype):
            # error: "str" has no attribute "tz"
            unit, tz = unit.unit, unit.tz  # type: ignore[union-attr]

        if unit != "ns":
            if isinstance(unit, str) and tz is None:
                # maybe a string like datetime64[ns, tz], which we support for
                # now.
                result = type(self).construct_from_string(unit)
                unit = result.unit
                tz = result.tz
                msg = (
                    f"Passing a dtype alias like 'datetime64[ns, {tz}]' "
                    "to DatetimeTZDtype is no longer supported. Use "
                    "'DatetimeTZDtype.construct_from_string()' instead."
                )
                raise ValueError(msg)
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("DatetimeTZDtype only supports s, ms, us, ns units")

        if tz:
            tz = timezones.maybe_get_tz(tz)
            tz = timezones.tz_standardize(tz)
        elif tz is not None:
            raise zoneinfo.ZoneInfoNotFoundError(tz)
        if tz is None:
            raise TypeError("A 'tz' is required.")

        self._unit = unit
        self._tz = tz

    @cache_readonly
    def _creso(self) -> int:
        # [FUNCTION DOC]
        # HELPER: cached. Returns the numpy datetime unit code for this dtype's unit
        # string.
        """
        The NPY_DATETIMEUNIT corresponding to this dtype's resolution.
        """
        return abbrev_to_npy_unit(self.unit)

    @property
    def unit(self) -> TimeUnit:
        # [FUNCTION DOC]
        # Property returning the unit string.
        """
        The precision of the datetime data.

        See Also
        --------
        DatetimeTZDtype.tz : Retrieves the timezone.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo("America/Los_Angeles"))
        >>> dtype.unit
        'ns'
        """
        return self._unit

    @property
    def tz(self) -> tzinfo:
        # [FUNCTION DOC]
        # Property returning the time zone object.
        """
        The timezone.

        See Also
        --------
        DatetimeTZDtype.unit : Retrieves precision of the datetime data.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo("America/Los_Angeles"))
        >>> dtype.tz
        zoneinfo.ZoneInfo(key='America/Los_Angeles')
        """
        return self._tz

    def construct_array_type(self) -> type_t[DatetimeArray]:
        # [FUNCTION DOC]
        # Returns the DatetimeArray class (imported lazily).
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import DatetimeArray

        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        # [FUNCTION DOC]
        # Class method. Parameter: string of the form datetime64 (or the short code)
        # followed by the unit and time zone in brackets. Raises a type error if not a
        # string, if it does not match, or if building the dtype fails with a key, type
        # or value error (bad zone or unit). Returns the dtype.
        """
        Construct a DatetimeTZDtype from a string.

        Parameters
        ----------
        string : str
            The string alias for this DatetimeTZDtype.
            Should be formatted like ``datetime64[ns, <tz>]``,
            where ``<tz>`` is the timezone name.

        Examples
        --------
        >>> DatetimeTZDtype.construct_from_string("datetime64[ns, UTC]")
        datetime64[ns, UTC]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                unit = cast("TimeUnit", d["unit"])
                return cls(unit=unit, tz=d["tz"])
            except (KeyError, TypeError, ValueError) as err:
                # KeyError if maybe_get_tz tries and fails to get a
                #  zoneinfo timezone (actually zoneinfo.ZoneInfoNotFoundError).
                # TypeError if we pass a nonsense tz;
                # ValueError if we pass a unit other than "ns"
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self) -> str_type:
        # [FUNCTION DOC]
        # Returns datetime64 with the unit and time zone in brackets.
        return f"datetime64[{self.unit}, {self.tz}]"

    @property
    def name(self) -> str_type:
        # [FUNCTION DOC]
        # Property; same as the string form.
        """A string representation of the dtype."""
        return str(self)

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash of the string form (marked in the source as in need of update).
        # make myself hashable
        # TODO: update this.
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # A string compares equal to the name, after rewriting a leading short code into
        # the long name. Otherwise equal only when other is a DatetimeTZDtype with the
        # same unit and zones comparing equal as time zones (so different zone object
        # types for one zone match).
        if isinstance(other, str):
            if other.startswith("M8["):
                other = f"datetime64[{other[3:]}"
            return other == self.name

        return (
            isinstance(other, DatetimeTZDtype)
            and self.unit == other.unit
            and tz_compare(self.tz, other.tz)
        )

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> DatetimeArray:
        # [FUNCTION DOC]
        # Converts a pyarrow array or chunked array to a DatetimeArray. Parameter:
        # array. Casts it safely to a timestamp type with this dtype's unit (the time
        # zone of the arrow type is dropped), converts to numpy (without requiring zero
        # copy for plain arrays) and wraps the result in a datetime array with this
        # dtype. Imports pyarrow lazily.
        """
        Construct DatetimeArray from pyarrow Array/ChunkedArray.

        Note: If the units in the pyarrow Array are the same as this
        DatetimeDtype, then values corresponding to the integer representation
        of ``NaT`` (e.g. one nanosecond before :attr:`pandas.Timestamp.min`)
        are converted to ``NaT``, regardless of the null indicator in the
        pyarrow array.

        Parameters
        ----------
        array : pyarrow.Array or pyarrow.ChunkedArray
            The Arrow array to convert to DatetimeArray.

        Returns
        -------
        extension array : DatetimeArray
        """
        import pyarrow

        from pandas.core.arrays import DatetimeArray

        array = array.cast(pyarrow.timestamp(unit=self._unit), safe=True)

        if isinstance(array, pyarrow.Array):
            np_arr = array.to_numpy(zero_copy_only=False)
        else:
            np_arr = array.to_numpy()

        return DatetimeArray._simple_new(np_arr, dtype=self)

    def __setstate__(self, state) -> None:
        # [FUNCTION DOC]
        # Unpickling support: restores the private time zone and unit from the state
        # mapping keys. Raises a key error if either is missing.
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._tz = state["tz"]
        self._unit = state["unit"]

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: common-type hook. If every dtype is timezone-aware with the same zone,
        # returns a dtype with the same zone and the finest unit among self and the
        # dtypes; otherwise defers to the base implementation.
        if all(isinstance(t, DatetimeTZDtype) and t.tz == self.tz for t in dtypes):
            np_dtype = np.max([cast(DatetimeTZDtype, t).base for t in [self, *dtypes]])
            unit = np.datetime_data(np_dtype)[0]
            unit = cast("TimeUnit", unit)
            return type(self)(unit=unit, tz=self.tz)
        return super()._get_common_dtype(dtypes)

    @cache_readonly
    def index_class(self) -> type_t[DatetimeIndex]:
        # [FUNCTION DOC]
        # Cached. Returns the datetime index class (imported lazily).
        from pandas import DatetimeIndex

        return DatetimeIndex


@register_extension_dtype
@set_module("pandas")
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    # [CLASS DOC]
    # Extension dtype for Period data, duck-typing numpy. Inherits from the compiled
    # period dtype base (which stores the frequency code and multiple) and the shared
    # pandas dtype base. Parameter: freq. State: the offset object (private freq) plus
    # the compiled base's code. Scalar type is Period, kind code is object, NaT is the
    # missing value, supports 2D and fast transpose. A class-level cache maps offsets to
    # period codes. Instances are created in the allocation hook, not in an initializer.
    # Equality is by the base class plus string-name shortcuts; hashing uses the
    # compiled base's hash. Pairs with the period array and period index.
    """
    An ExtensionDtype for Period data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    freq : str or DateOffset
        The frequency of this PeriodDtype.

    Attributes
    ----------
    freq

    Methods
    -------
    None

    See Also
    --------
    Period : Represents a single time period.
    PeriodIndex : Immutable index for period data.
    date_range : Return a fixed frequency DatetimeIndex.
    Series : One-dimensional array with axis labels.
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Examples
    --------
    >>> pd.PeriodDtype(freq="D")
    period[D]

    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())
    period[M]
    """

    type: type[Period] = Period
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    num = 102
    _metadata = ("freq",)
    _match = re.compile(r"(P|p)eriod\[(?P<freq>.+)\]")
    # error: Incompatible types in assignment (expression has type
    # "Dict[int, PandasExtensionDtype]", base class "PandasExtensionDtype"
    # defined the type as "Dict[str, PandasExtensionDtype]")  [assignment]
    _cache_dtypes: dict[BaseOffset, int] = {}  # type: ignore[assignment]
    __hash__ = PeriodDtypeBase.__hash__
    _freq: BaseOffset
    _supports_2d = True
    _can_fast_transpose = True

    def __new__(cls, freq) -> PeriodDtype:  # noqa: PYI034
        # [FUNCTION DOC]
        # Parameter: freq, a PeriodDtype (returned unchanged), an offset, or a
        # string/other value parsed strictly into an offset (raises a type error for
        # unparsable input). A business-day frequency emits a future warning
        # (deprecated). The offset's period code is looked up in or added to the class
        # cache, then a new instance is created from the code and the offset's multiple,
        # and the offset stored. Returns the new dtype. Side effect: mutates the
        # class-level cache.
        """
        Parameters
        ----------
        freq : PeriodDtype, BaseOffset, or string
        """
        if isinstance(freq, PeriodDtype):
            return freq

        if not isinstance(freq, BaseOffset):
            freq = cls._parse_dtype_strict(freq)

        if isinstance(freq, BDay):
            # GH#53446
            # TODO(3.0): enforcing this will close GH#10575
            warnings.warn(
                "PeriodDtype[B] is deprecated and will be removed in a future "
                "version. Use a DatetimeIndex with freq='B' instead",
                FutureWarning,  # pdlint: ignore[warning_class]
                stacklevel=find_stack_level(),
            )

        try:
            dtype_code = cls._cache_dtypes[freq]
        except KeyError:
            dtype_code = freq._period_dtype_code
            cls._cache_dtypes[freq] = dtype_code
        u = PeriodDtypeBase.__new__(cls, dtype_code, freq.n)
        u._freq = freq
        return u

    def __reduce__(self) -> tuple[type_t[Self], tuple[str_type]]:
        # [FUNCTION DOC]
        # Pickle support: returns the class and a one-element argument tuple containing
        # the dtype name, so unpickling reconstructs from the string form.
        return type(self), (self.name,)

    @property
    def freq(self) -> BaseOffset:
        # [FUNCTION DOC]
        # Property returning the frequency offset object.
        """
        The frequency object of this PeriodDtype.

        The `freq` property returns the `BaseOffset` object that represents the
        frequency of the PeriodDtype. This frequency specifies the interval (e.g.,
        daily, monthly, yearly) associated with the Period type. It is essential
        for operations that depend on time-based calculations within a period index
        or series.

        See Also
        --------
        Period : Represents a period of time.
        PeriodIndex : Immutable ndarray holding ordinal values indicating
            regular periods.
        PeriodDtype : An ExtensionDtype for Period data.
        date_range : Return a fixed frequency range of dates.

        Examples
        --------
        >>> dtype = pd.PeriodDtype(freq="D")
        >>> dtype.freq
        <Day>
        """
        return self._freq

    @classmethod
    def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset:
        # [FUNCTION DOC]
        # HELPER: class method. Parameter: freq, a string. Strings beginning with the
        # period bracket form (either capitalization) have the inner frequency
        # extracted; then the string is converted to an offset for period use. Returns
        # the offset. Raises a type error when the argument is not a string or no offset
        # results.
        # KNOWN ISSUE: an inline comment claims freq is already a string, yet the code
        # first checks that it is one and raises a type error otherwise, so non-string
        # input is expected.
        if isinstance(freq, str):  # note: freq is already of type str!
            if freq.startswith(("Period[", "period[")):
                m = cls._match.search(freq)
                if m is not None:
                    freq = m.group("freq")

            freq_offset = to_offset(freq, is_period=True)
            if freq_offset is not None:
                return freq_offset

        raise TypeError(
            "PeriodDtype argument should be string or BaseOffset, "
            f"got {type(freq).__name__}"
        )

    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        # [FUNCTION DOC]
        # Class method. Succeeds only for strings starting with the period bracket form
        # (either capitalization) or for offset objects, constructing via the freq
        # argument; a value error during construction is swallowed. Otherwise raises a
        # type error (with a message about expecting a string if the input is not a
        # string).
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        if (
            isinstance(string, str) and (string.startswith(("period[", "Period[")))
        ) or isinstance(string, BaseOffset):
            # do not parse string like U as period[U]
            # avoid tuple to be regarded as freq
            try:
                return cls(freq=string)
            except ValueError:
                pass
        if isinstance(string, str):
            msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        else:
            msg = f"'construct_from_string' expects a string, got {type(string)}"
        raise TypeError(msg)

    def __str__(self) -> str_type:
        # [FUNCTION DOC]
        # Returns the name.
        return self.name

    @property
    def name(self) -> str_type:
        # [FUNCTION DOC]
        # Property. Returns the lowercase period word with the frequency string in
        # brackets.
        return f"period[{self._freqstr}]"

    @property
    def na_value(self) -> NaTType:
        # [FUNCTION DOC]
        # Property returning the NaT singleton.
        return NaT

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # A string compares equal after lowering its first character, to the name. Other
        # objects defer to the parent comparison (class and metadata).
        if isinstance(other, str):
            return other[:1].lower() + other[1:] == self.name

        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Negation of equality.
        return not self.__eq__(other)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        # [FUNCTION DOC]
        # Class method. Parameter: dtype. Strings only match when they use the period
        # bracket form (either capitalization) and parse strictly (a value error means
        # no match); other strings are never dtypes (bare frequency strings are rejected
        # deliberately). Everything else uses the base class test.
        """
        Return a boolean if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            # PeriodDtype can be instantiated from freq string like "U",
            # but doesn't regard freq str like "U" as dtype.
            if dtype.startswith(("period[", "Period[")):
                try:
                    return cls._parse_dtype_strict(dtype) is not None
                except ValueError:
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def construct_array_type(self) -> type_t[PeriodArray]:
        # [FUNCTION DOC]
        # Returns the PeriodArray class (imported lazily).
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import PeriodArray

        return PeriodArray

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> PeriodArray:
        # [FUNCTION DOC]
        # Converts a pyarrow array or chunked array to a PeriodArray. Parameter: array.
        # Each chunk is converted to a 64-bit integer array plus a null mask (via a
        # helper), wrapped in a period array with this dtype, and masked entries are set
        # to NaT. Chunks are concatenated; zero chunks produce an empty period array.
        """
        Construct PeriodArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays import PeriodArray
        from pandas.core.arrays.arrow._arrow_utils import (
            pyarrow_array_to_numpy_and_mask,
        )

        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        results = []
        for arr in chunks:
            data, mask = pyarrow_array_to_numpy_and_mask(arr, dtype=np.dtype(np.int64))
            parr = PeriodArray(data.copy(), dtype=self, copy=False)
            # error: Invalid index type "ndarray[Any, dtype[bool_]]" for "PeriodArray";
            # expected type "Union[int, Sequence[int], Sequence[bool], slice]"
            parr[~mask] = NaT  # type: ignore[index]
            results.append(parr)

        if not results:
            return PeriodArray(np.array([], dtype="int64"), dtype=self, copy=False)
        return PeriodArray._concat_same_type(results)

    @cache_readonly
    def index_class(self) -> type_t[PeriodIndex]:
        # [FUNCTION DOC]
        # Cached. Returns the period index class (imported lazily).
        from pandas import PeriodIndex

        return PeriodIndex


@register_extension_dtype
@set_module("pandas")
class IntervalDtype(PandasExtensionDtype):
    # [CLASS DOC]
    # Extension dtype for Interval data, duck-typing numpy. Parameters: subtype (a
    # dtype, name or another IntervalDtype) and closed (one of right, left, both,
    # neither, or none). State: private subtype (numpy dtype or none when uninitialized)
    # and closed (or none, for partially initialized). Identity metadata: subtype and
    # closed. Name is interval; scalar type is Interval; kind code is object. Category,
    # object and string subtypes are rejected. Paired with the interval array and
    # interval index.
    """
    An ExtensionDtype for Interval data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.
    closed : {'right', 'left', 'both', 'neither'}, default 'right'
        Whether the interval is closed on the left-side, right-side, both or
        neither. See the Notes for more detailed explanation.

    Attributes
    ----------
    subtype

    Methods
    -------
    None

    See Also
    --------
    PeriodDtype : An ExtensionDtype for Period data.

    Examples
    --------
    >>> pd.IntervalDtype(subtype="int64", closed="both")
    interval[int64, both]
    """

    name = "interval"
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    num = 103
    _metadata = (
        "subtype",
        "closed",
    )

    _match = re.compile(
        r"(I|i)nterval\[(?P<subtype>[^,]+(\[.+\])?)"
        r"(, (?P<closed>(right|left|both|neither)))?\]"
    )

    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _subtype: None | np.dtype
    _closed: IntervalClosedType | None

    def __init__(self, subtype=None, closed: IntervalClosedType | None = None) -> None:
        # [FUNCTION DOC]
        # Parameters: subtype (default none), closed (default none). An invalid closed
        # value raises a value error. An IntervalDtype subtype is copied (a conflicting
        # closed raises a value error). None or the bare word interval gives an
        # uninitialized subtype and uses the closed argument. Otherwise a string may be
        # parsed with a pattern to extract the subtype and a closed value (a mismatch
        # with the keyword raises a value error); the subtype is converted to a dtype
        # (failure becomes a type error); categorical, object and string subtypes raise
        # a type error.
        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )

        if closed is not None and closed not in {"right", "left", "both", "neither"}:
            raise ValueError("closed must be one of 'right', 'left', 'both', 'neither'")

        if isinstance(subtype, IntervalDtype):
            if closed is not None and closed != subtype.closed:
                raise ValueError(
                    "dtype.closed and 'closed' do not match. "
                    "Try IntervalDtype(dtype.subtype, closed) instead."
                )
            self._subtype = subtype._subtype
            self._closed = subtype._closed
        elif subtype is None:
            # we are called as an empty constructor
            # generally for pickle compat
            self._subtype = None
            self._closed = closed
        elif isinstance(subtype, str) and subtype.lower() == "interval":
            self._subtype = None
            self._closed = closed
        else:
            if isinstance(subtype, str):
                m = IntervalDtype._match.search(subtype)
                if m is not None:
                    gd = m.groupdict()
                    subtype = gd["subtype"]
                    if gd.get("closed", None) is not None:
                        if closed is not None:
                            if closed != gd["closed"]:
                                raise ValueError(
                                    "'closed' keyword does not match value "
                                    "specified in dtype string"
                                )
                        closed = gd["closed"]  # type: ignore[assignment]

            try:
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError("could not construct IntervalDtype") from err
            if CategoricalDtype.is_dtype(subtype) or is_string_dtype(subtype):
                # GH 19016
                msg = (
                    "category, object, and string subtypes are not supported "
                    "for IntervalDtype"
                )
                raise TypeError(msg)
            self._subtype = subtype
            self._closed = closed

    @cache_readonly
    def _can_hold_na(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached. Returns false for integer and unsigned subtypes and true
        # otherwise. Raises a not-implemented error if the subtype is not yet set.
        subtype = self._subtype
        if subtype is None:
            # partially-initialized
            raise NotImplementedError(
                "_can_hold_na is not defined for partially-initialized IntervalDtype"
            )
        if subtype.kind in "iu":
            return False
        return True

    @property
    def closed(self) -> IntervalClosedType:
        # [FUNCTION DOC]
        # Property returning the closed setting (possibly none).
        return self._closed  # type: ignore[return-value]

    @property
    def subtype(self):
        # [FUNCTION DOC]
        # Property returning the subtype dtype (possibly none).
        """
        The dtype of the Interval bounds.

        See Also
        --------
        IntervalDtype: An ExtensionDtype for Interval data.

        Examples
        --------
        >>> dtype = pd.IntervalDtype(subtype="int64", closed="both")
        >>> dtype.subtype
        dtype('int64')
        """
        return self._subtype

    def construct_array_type(self) -> type[IntervalArray]:
        # [FUNCTION DOC]
        # Returns the IntervalArray class (imported lazily).
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import IntervalArray

        return IntervalArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> IntervalDtype:
        # [FUNCTION DOC]
        # Class method. Parameter: string. Raises a type error if not a string or if it
        # is neither the bare word interval (any case) nor matching the interval bracket
        # pattern; otherwise returns a dtype built from the string.
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string.lower() == "interval" or cls._match.search(string) is not None:
            return cls(string)

        msg = (
            f"Cannot construct a 'IntervalDtype' from '{string}'.\n\n"
            "Incorrectly formatted string passed to constructor. "
            "Valid formats include Interval or Interval[dtype] "
            "where dtype is numeric, datetime, or timedelta"
        )
        raise TypeError(msg)

    @property
    def type(self) -> type[Interval]:
        # [FUNCTION DOC]
        # Property returning the Interval scalar class.
        return Interval

    def __str__(self) -> str_type:
        # [FUNCTION DOC]
        # Returns the bare word if the subtype is unset; the word with the subtype in
        # brackets if only closed is unset; otherwise the subtype and closed in
        # brackets.
        if self.subtype is None:
            return "interval"
        if self.closed is None:
            # Only partially initialized GH#38394
            return f"interval[{self.subtype}]"
        return f"interval[{self.subtype}, {self.closed}]"

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash of the string form.
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Strings compare case-insensitively against the plain name or the full string
        # form. Non-IntervalDtype objects are unequal. If either side lacks a subtype
        # they are equal (an uninitialized dtype matches any interval dtype, regardless
        # of closed). Otherwise equal when closed settings and subtypes both match.
        if isinstance(other, str):
            return other.lower() in (self.name.lower(), str(self).lower())
        elif not isinstance(other, IntervalDtype):
            return False
        elif self.subtype is None or other.subtype is None:
            # None should match any subtype
            return True
        elif self.closed != other.closed:
            return False
        else:
            return self.subtype == other.subtype

    def __setstate__(self, state) -> None:
        # [FUNCTION DOC]
        # Unpickling support: restores the subtype from the state (required key) and
        # closed from the state (default none for older pickles; the key is removed from
        # the mapping).
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._subtype = state["subtype"]

        # backward-compat older pickles won't have "closed" key
        self._closed = state.pop("closed", None)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        # [FUNCTION DOC]
        # Class method. Strings match only if they start (case-insensitively) with the
        # word interval and parse through the string constructor (value and type errors
        # mean no match); other strings are never dtypes. Anything else uses the base
        # class test.
        """
        Return a boolean if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            if dtype.lower().startswith("interval"):
                try:
                    return cls.construct_from_string(dtype) is not None
                except (ValueError, TypeError):
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> IntervalArray:
        # [FUNCTION DOC]
        # Converts a pyarrow array or chunked array of interval-like structs to an
        # IntervalArray. Parameter: array. For each chunk (unwrapping arrow extension
        # arrays to their storage) takes the left and right fields as arrays of this
        # subtype and builds an interval array with this dtype's closed setting; chunks
        # are concatenated; with zero chunks an empty interval array is produced.
        """
        Construct IntervalArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays import IntervalArray

        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        results = []
        for arr in chunks:
            if isinstance(arr, pyarrow.ExtensionArray):
                arr = arr.storage
            left = np.asarray(arr.field("left"), dtype=self.subtype)
            right = np.asarray(arr.field("right"), dtype=self.subtype)
            iarr = IntervalArray.from_arrays(left, right, closed=self.closed)
            results.append(iarr)

        if not results:
            return IntervalArray.from_arrays(
                np.array([], dtype=self.subtype),
                np.array([], dtype=self.subtype),
                closed=self.closed,
            )
        return IntervalArray._concat_same_type(results)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: common-type hook. Returns none unless all dtypes are interval dtypes.
        # If their closed settings differ returns numpy object dtype. Otherwise finds
        # the common type of the subtypes; object gives numpy object dtype, else an
        # interval dtype with that subtype and the shared closed setting.
        if not all(isinstance(x, IntervalDtype) for x in dtypes):
            return None

        closed = cast("IntervalDtype", dtypes[0]).closed
        if not all(cast("IntervalDtype", x).closed == closed for x in dtypes):
            return np.dtype(object)

        from pandas.core.dtypes.cast import find_common_type

        common = find_common_type([cast("IntervalDtype", x).subtype for x in dtypes])
        if common == object:
            return np.dtype(object)
        return IntervalDtype(common, closed=closed)

    @cache_readonly
    def index_class(self) -> type_t[IntervalIndex]:
        # [FUNCTION DOC]
        # Cached. Returns the interval index class (imported lazily).
        from pandas import IntervalIndex

        return IntervalIndex


class NumpyEADtype(ExtensionDtype):
    # [CLASS DOC]
    # HELPER: internal extension dtype that wraps a numpy dtype so numpy-backed data can
    # be handled as extension arrays (mainly for the numpy extension array). Parameter:
    # dtype (anything convertible to a numpy dtype, or another instance). State: the
    # wrapped numpy dtype under a private name that is also the metadata for equality
    # and hashing. Forwards name, type, kind and item size from the wrapped dtype;
    # numeric and boolean flags derived from kind; no 2D support.
    """
    A Pandas ExtensionDtype for NumPy dtypes.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    dtype : object
        Object to be converted to a NumPy data type object.

    See Also
    --------
    numpy.dtype
    """

    _metadata = ("_dtype",)
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, dtype: npt.DTypeLike | NumpyEADtype | None) -> None:
        # [FUNCTION DOC]
        # Parameter: dtype, an object convertible to a numpy dtype, a NumpyEADtype
        # (unwrapped, making construction idempotent) or none (numpy treats none as
        # float64). Stores the numpy dtype. Raises whatever numpy raises for invalid
        # input.
        if isinstance(dtype, NumpyEADtype):
            # make constructor idempotent
            dtype = dtype.numpy_dtype
        self._dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the class name with the quoted name of the wrapped dtype in
        # parentheses.
        return f"NumpyEADtype({self.name!r})"

    @property
    def numpy_dtype(self) -> np.dtype:
        # [FUNCTION DOC]
        # Property returning the wrapped numpy dtype.
        """
        The NumPy dtype this NumpyEADtype wraps.
        """
        return self._dtype

    @property
    def name(self) -> str:
        # [FUNCTION DOC]
        # Property returning the name of the wrapped numpy dtype.
        """
        A bit-width name for this data-type.
        """
        return self._dtype.name

    @property
    def type(self) -> type[np.generic]:
        # [FUNCTION DOC]
        # Property returning the scalar type of the wrapped numpy dtype.
        """
        The type object used to instantiate a scalar of this NumPy data-type.
        """
        return self._dtype.type

    @property
    def _is_numeric(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true for boolean, signed, unsigned, float and complex kinds;
        # false for object, string, unicode, void and datetime-like.
        # exclude object, str, unicode, void.
        return self.kind in set("biufc")

    @property
    def _is_boolean(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true when the wrapped kind is boolean.
        return self.kind == "b"

    @classmethod
    def construct_from_string(cls, string: str) -> NumpyEADtype:
        # [FUNCTION DOC]
        # Class method. Parameter: string. Builds the numpy dtype from it and wraps it.
        # Numpy type errors are re-raised as type errors with a message that depends on
        # whether the input was a string.
        try:
            dtype = np.dtype(string)
        except TypeError as err:
            if not isinstance(string, str):
                msg = f"'construct_from_string' expects a string, got {type(string)}"
            else:
                msg = f"Cannot construct a 'NumpyEADtype' from '{string}'"
            raise TypeError(msg) from err
        return cls(dtype)

    def construct_array_type(self) -> type_t[NumpyExtensionArray]:
        # [FUNCTION DOC]
        # Returns the NumpyExtensionArray class (imported lazily).
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import NumpyExtensionArray

        return NumpyExtensionArray

    @property
    def kind(self) -> str:
        # [FUNCTION DOC]
        # Property returning the kind code of the wrapped numpy dtype.
        """
        A character code (one of 'biufcmMOSUV') identifying the general kind of data.
        """
        return self._dtype.kind

    @property
    def itemsize(self) -> int:
        # [FUNCTION DOC]
        # Property returning the element size in bytes of the wrapped dtype.
        """
        The element size of this data-type object.
        """
        return self._dtype.itemsize


class BaseMaskedDtype(ExtensionDtype):
    # [CLASS DOC]
    # HELPER: internal base class for the nullable (mask-based) dtypes of the arrays
    # package: integers, floats and booleans. Subclasses must define the scalar type and
    # an internal fill value. Provides: NA singleton as the missing value; the numpy
    # dtype derived (cached) from the scalar type; kind and item size cached from it;
    # fill values used for any/all reductions; conversion from a numpy dtype; and
    # common-type resolution.
    """
    Base class for dtypes for BaseMaskedArray subclasses.
    """

    base: DtypeObj | None = None
    type: type
    _internal_fill_value: Scalar

    @property
    def _truthy_value(self):
        # [FUNCTION DOC]
        # HELPER: property. Fill value used in "any" reductions: one for float kinds (as
        # a float) and integer kinds, true otherwise.
        # Fill values used for 'any'
        if self.kind == "f":
            return 1.0
        if self.kind in "iu":
            return 1
        return True

    @property
    def _falsey_value(self):
        # [FUNCTION DOC]
        # HELPER: property. Fill value used in "all" reductions: zero for float kinds
        # (as a float) and integer kinds, false otherwise.
        # Fill values used for 'all'
        if self.kind == "f":
            return 0.0
        if self.kind in "iu":
            return 0
        return False

    @property
    def na_value(self) -> libmissing.NAType:
        # [FUNCTION DOC]
        # Property returning the NA singleton.
        return libmissing.NA

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        # [FUNCTION DOC]
        # Cached. Returns the numpy dtype built from the subclass's scalar type.
        """Return an instance of our numpy dtype"""
        return np.dtype(self.type)

    @cache_readonly
    def kind(self) -> str:
        # [FUNCTION DOC]
        # Cached. Returns the kind code of the numpy dtype.
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        # [FUNCTION DOC]
        # Cached. Returns the numpy dtype's item size in bytes.
        """Return the number of bytes in this dtype"""
        return self.numpy_dtype.itemsize

    def construct_array_type(self) -> type_t[BaseMaskedArray]:
        # [FUNCTION DOC]
        # Abstract. Raises a not-implemented error in this base; subclasses return their
        # masked array class.
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        raise NotImplementedError

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> BaseMaskedDtype:
        # [FUNCTION DOC]
        # Class method. Parameter: dtype, a numpy dtype. Returns the matching nullable
        # dtype instance: boolean kind gives a new boolean masked dtype,
        # integer/unsigned kinds look up the integer mapping table, float kinds the
        # float table (lazy imports). Raises a not-implemented error for other kinds, or
        # a key error if the table lacks the exact numpy dtype.
        """
        Construct the MaskedDtype corresponding to the given numpy dtype.
        """
        if dtype.kind == "b":
            from pandas.core.arrays.boolean import BooleanDtype

            return BooleanDtype()
        elif dtype.kind in "iu":
            from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE

            return NUMPY_INT_TO_DTYPE[dtype]
        elif dtype.kind == "f":
            from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE

            return NUMPY_FLOAT_TO_DTYPE[dtype]
        else:
            raise NotImplementedError(dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: common-type hook. Replaces masked dtypes by their numpy dtypes, finds
        # the generic common type, and re-wraps it as this class's masked dtype via the
        # numpy-dtype constructor. Returns none if the common type is not numpy or
        # cannot be represented (key or not-implemented errors).
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        from pandas.core.dtypes.cast import find_common_type

        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, BaseMaskedDtype) else dtype
                for dtype in dtypes
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            # If we ever support e.g. Masked[DatetimeArray] then this will change
            return None
        try:
            return type(self).from_numpy_dtype(new_dtype)
        except (KeyError, NotImplementedError):
            return None


@register_extension_dtype
@set_module("pandas")
class SparseDtype(ExtensionDtype):
    # [CLASS DOC]
    # Extension dtype for data stored in a SparseArray: the dtype of the non-fill values
    # plus a fill value that is not stored. Parameters: dtype (default float64; a
    # string, numpy dtype, type or another SparseDtype) and fill_value (default none
    # meaning the natural missing value of the subtype, e.g. NaN for floats, zero for
    # integers). State: private numpy dtype and fill value. Immutable flag is set.
    # Identity metadata includes a NA-flag so that NaN and zero float fill values hash
    # differently. Only numpy subtypes are allowed (text becomes object). Name takes the
    # form Sparse bracket subtype and fill value. Paired with the sparse array.
    """
    Dtype for data stored in :class:`SparseArray`.

    ``SparseDtype`` is used as the data type for :class:`SparseArray`, enabling
    more efficient storage of data that contains a significant number of
    repetitive values typically represented by a fill value. It supports any
    scalar dtype as the underlying data type of the non-fill values.

    Parameters
    ----------
    dtype : str, ExtensionDtype, numpy.dtype, type, default numpy.float64
        The dtype of the underlying array storing the non-fill value values.
    fill_value : scalar, optional
        The scalar value not stored in the SparseArray. By default, this
        depends on ``dtype``.

        =========== ==========
        dtype       na_value
        =========== ==========
        float       ``np.nan``
        complex     ``np.nan``
        int         ``0``
        bool        ``False``
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The default value may be overridden by specifying a ``fill_value``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    arrays.SparseArray : The array structure that uses SparseDtype
        for data representation.

    Examples
    --------
    >>> ser = pd.Series([1, 0, 0], dtype=pd.SparseDtype(dtype=int, fill_value=0))
    >>> ser
    0    1
    1    0
    2    0
    dtype: Sparse[int64, 0]
    >>> ser.sparse.density
    0.3333333333333333
    """

    _is_immutable = True

    # We include `_is_na_fill_value` in the metadata to avoid hash collisions
    # between SparseDtype(float, 0.0) and SparseDtype(float, nan).
    # Without is_na_fill_value in the comparison, those would be equal since
    # hash(nan) is (sometimes?) 0.
    _metadata = ("_dtype", "_fill_value", "_is_na_fill_value")

    def __init__(self, dtype: Dtype = np.float64, fill_value: Any = None) -> None:
        # [FUNCTION DOC]
        # Parameters: dtype (default float64), fill_value (default none). Another
        # SparseDtype contributes its subtype and, when fill_value is none, its fill
        # value. The dtype is normalized; string dtypes become object; non-numpy dtypes
        # raise a type error. A none fill_value is replaced by the NA value of the
        # dtype. Stores both and validates the fill value (which may raise a value
        # error).
        if isinstance(dtype, type(self)):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )
        from pandas.core.dtypes.missing import na_value_for_dtype

        dtype = pandas_dtype(dtype)
        if is_string_dtype(dtype):
            dtype = np.dtype("object")
        if not isinstance(dtype, np.dtype):
            # GH#53160
            raise TypeError("SparseDtype subtype must be a numpy dtype")

        if fill_value is None:
            fill_value = na_value_for_dtype(dtype)

        self._dtype = dtype
        self._fill_value = fill_value
        self._check_fill_value()

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Delegates to the parent hash (restated because defining equality removes
        # inherited hashing).
        # Python3 doesn't inherit __hash__ when a base class overrides
        # __eq__, so we explicitly do it here.
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Strings are parsed through the string constructor (failure gives false). For
        # SparseDtype operands: subtypes must be equal; if either fill value is missing,
        # fill values match when one's type is an instance of the other's type (so any
        # float NaN equal, but NaN differs from NaT); otherwise fill values are compared
        # by equality with a numpy comparison warning suppressed. Other types are
        # unequal.
        # We have to override __eq__ to handle NA values in _metadata.
        # The base class does simple == checks, which fail for NA.
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False

        if isinstance(other, type(self)):
            subtype = self.subtype == other.subtype
            if self._is_na_fill_value or other._is_na_fill_value:
                # this case is complicated by two things:
                # SparseDtype(float, float(nan)) == SparseDtype(float, np.nan)
                # SparseDtype(float, np.nan)     != SparseDtype(float, pd.NaT)
                # i.e. we want to treat any floating-point NaN as equal, but
                # not a floating-point NaN and a datetime NaT.
                fill_value = isinstance(
                    self.fill_value, type(other.fill_value)
                ) or isinstance(other.fill_value, type(self.fill_value))
            else:
                with warnings.catch_warnings():
                    # Ignore spurious numpy warning
                    warnings.filterwarnings(
                        "ignore",
                        "elementwise comparison failed",
                        category=DeprecationWarning,
                    )

                    fill_value = self.fill_value == other.fill_value

            return subtype and fill_value
        return False

    @property
    def fill_value(self):
        # [FUNCTION DOC]
        # Property returning the stored fill value (the value not stored in the sparse
        # array).
        """
        The fill value of the array.

        Converting the SparseArray to a dense ndarray will fill the
        array with this value.

        .. warning::

           It's possible to end up with a SparseArray that has ``fill_value``
           values in ``sp_values``. This can occur, for example, when setting
           ``SparseArray.fill_value`` directly.
        """
        return self._fill_value

    def _check_fill_value(self) -> None:
        # [FUNCTION DOC]
        # HELPER: validates the fill value. Raises a value error if it is not a scalar.
        # A missing fill value must be a valid missing marker for the subtype; a
        # non-missing one must be holdable by an empty array of the subtype
        # (datetime-likes wrapped appropriately); otherwise a value error says the fill
        # value must be valid for the subtype.
        if not lib.is_scalar(self._fill_value):
            raise ValueError(
                f"fill_value must be a scalar. Got {self._fill_value} instead"
            )

        from pandas.core.dtypes.cast import can_hold_element
        from pandas.core.dtypes.missing import (
            is_valid_na_for_dtype,
            isna,
        )

        from pandas.core.construction import ensure_wrapped_if_datetimelike

        # GH#23124 require fill_value and subtype to match
        val = self._fill_value
        if isna(val):
            if not is_valid_na_for_dtype(val, self.subtype):
                raise ValueError(
                    # GH#53043
                    "fill_value must be a valid value for the SparseDtype.subtype"
                )
        else:
            dummy = np.empty(0, dtype=self.subtype)
            dummy = ensure_wrapped_if_datetimelike(dummy)

            if not can_hold_element(dummy, val):
                raise ValueError(
                    # GH#53043
                    "fill_value must be a valid value for the SparseDtype.subtype"
                )

    @property
    def _is_na_fill_value(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true if the fill value is missing.
        from pandas import isna

        return isna(self.fill_value)

    @property
    def _is_numeric(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true unless the subtype is object.
        return not self.subtype == object

    @property
    def _is_boolean(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true when the subtype kind is boolean.
        return self.subtype.kind == "b"

    @property
    def kind(self) -> str:
        # [FUNCTION DOC]
        # Property returning the kind code of the subtype.
        # KNOWN ISSUE: the docstring says the value is the sparse kind, either integer
        # or block, but the code returns the numpy kind code of the subtype.
        """
        The sparse kind. Either 'integer', or 'block'.
        """
        return self.subtype.kind

    @property
    def type(self):
        # [FUNCTION DOC]
        # Property returning the scalar type of the subtype.
        return self.subtype.type

    @property
    def subtype(self):
        # [FUNCTION DOC]
        # Property returning the numpy dtype of the non-fill values.
        return self._dtype

    @property
    def name(self) -> str:
        # [FUNCTION DOC]
        # Property. Returns the word Sparse with the subtype name and the repr of the
        # fill value in brackets.
        return f"Sparse[{self.subtype.name}, {self.fill_value!r}]"

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the name.
        return self.name

    def construct_array_type(self) -> type_t[SparseArray]:
        # [FUNCTION DOC]
        # Returns the SparseArray class (imported lazily).
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.sparse.array import SparseArray

        return SparseArray

    @classmethod
    def construct_from_string(cls, string: str) -> SparseDtype:
        # [FUNCTION DOC]
        # Class method. Parameter: string. Raises a type error if not a string, if it
        # does not start with the word Sparse, if parsing fails, or if the string
        # contains a fill value that is not the default for the parsed subtype (checked
        # by comparing with the string form of the built dtype). Accepts the bare word
        # (float64 subtype) and the bracket forms with a subtype and optional default
        # fill value. Returns the dtype built with the default fill value.
        """
        Construct a SparseDtype from a string form.

        Parameters
        ----------
        string : str
            Can take the following forms.

            string           dtype
            ================ ============================
            'int'            SparseDtype[np.int64, 0]
            'Sparse'         SparseDtype[np.float64, nan]
            'Sparse[int]'    SparseDtype[np.int64, 0]
            'Sparse[int, 0]' SparseDtype[np.int64, 0]
            ================ ============================

            It is not possible to specify non-default fill values
            with a string. An argument like ``'Sparse[int, 1]'``
            will raise a ``TypeError`` because the default fill value
            for integers is 0.

        Returns
        -------
        SparseDtype
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        msg = f"Cannot construct a 'SparseDtype' from '{string}'"
        if string.startswith("Sparse"):
            try:
                sub_type, has_fill_value = cls._parse_subtype(string)
            except ValueError as err:
                raise TypeError(msg) from err
            else:
                result = SparseDtype(sub_type)
                msg = (
                    f"Cannot construct a 'SparseDtype' from '{string}'.\n\nIt "
                    "looks like the fill_value in the string is not "
                    "the default for the dtype. Non-default fill_values "
                    "are not supported. Use the 'SparseDtype()' "
                    "constructor instead."
                )
                if has_fill_value and str(result) != string:
                    raise TypeError(msg)
                return result
        else:
            raise TypeError(msg)

    @staticmethod
    def _parse_subtype(dtype: str) -> tuple[str, bool]:
        # [FUNCTION DOC]
        # HELPER: static method. Parameter: dtype, a string. Uses a pattern to extract
        # the subtype text and whether a fill value was present; the bare word gives
        # float64 with no fill value; anything else raises a value error. Returns the
        # pair of subtype text and the boolean.
        """
        Parse a string to get the subtype

        Parameters
        ----------
        dtype : str
            A string like

            * Sparse[subtype]
            * Sparse[subtype, fill_value]

        Returns
        -------
        subtype : str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted.
        """
        xpr = re.compile(r"Sparse\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\]$")
        m = xpr.match(dtype)
        has_fill_value = False
        if m:
            subtype = m.groupdict()["subtype"]
            has_fill_value = bool(m.groupdict()["fill_value"])
        elif dtype == "Sparse":
            subtype = "float64"
        else:
            raise ValueError(f"Cannot parse {dtype}")
        return subtype, has_fill_value

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        # [FUNCTION DOC]
        # Class method. Takes the dtype attribute if present. Strings starting with the
        # word Sparse are parsed to a numpy dtype of the subtype (parse errors propagate
        # as value errors); instances of the class return true. Finally returns true for
        # numpy dtypes and for the bare string.
        # KNOWN ISSUE: apart from sparse dtype instances, any numpy dtype is reported as
        # a match, which is broader than the name of the method suggests.
        dtype = getattr(dtype, "dtype", dtype)
        if isinstance(dtype, str) and dtype.startswith("Sparse"):
            sub_type, _ = cls._parse_subtype(dtype)
            dtype = np.dtype(sub_type)
        elif isinstance(dtype, cls):
            return True
        return isinstance(dtype, np.dtype) or dtype == "Sparse"

    def update_dtype(self, dtype) -> SparseDtype:
        # [FUNCTION DOC]
        # Parameter: dtype, a string, numpy dtype or SparseDtype. A SparseDtype is
        # returned as is. A numpy dtype (after normalization) gives a new SparseDtype
        # whose fill value is the current fill value cast to that dtype. Extension
        # dtypes raise a type error. A fill value that cannot be converted raises the
        # underlying error (documented as a value error).
        """
        Convert the SparseDtype to a new dtype.

        This takes care of converting the ``fill_value``.

        Parameters
        ----------
        dtype : Union[str, numpy.dtype, SparseDtype]
            The new dtype to use.

            * For a SparseDtype, it is simply returned
            * For a NumPy dtype (or str), the current fill value
              is converted to the new dtype, and a SparseDtype
              with `dtype` and the new fill value is returned.

        Returns
        -------
        SparseDtype
            A new SparseDtype with the correct `dtype` and fill value
            for that `dtype`.

        Raises
        ------
        ValueError
            When the current fill value cannot be converted to the
            new `dtype` (e.g. trying to convert ``np.nan`` to an
            integer dtype).


        Examples
        --------
        >>> SparseDtype(int, 0).update_dtype(float)
        Sparse[float64, np.float64(0.0)]

        >>> SparseDtype(int, 1).update_dtype(SparseDtype(float, np.nan))
        Sparse[float64, nan]
        """
        from pandas.core.dtypes.astype import astype_array
        from pandas.core.dtypes.common import pandas_dtype

        cls = type(self)
        dtype = pandas_dtype(dtype)

        if not isinstance(dtype, cls):
            if not isinstance(dtype, np.dtype):
                raise TypeError("sparse arrays of extension dtypes not supported")

            fv_asarray = np.atleast_1d(np.array(self.fill_value))
            fvarr = astype_array(fv_asarray, dtype)
            # NB: not fv_0d.item(), as that casts dt64->int
            fill_value = fvarr[0]
            dtype = cls(dtype, fill_value=fill_value)

        return dtype

    @property
    def _subtype_with_str(self):
        # [FUNCTION DOC]
        # HELPER: property. Returns the string type when the fill value is a string (so
        # string data can be distinguished within object storage); otherwise the
        # subtype.
        """
        Whether the SparseDtype's subtype should be considered ``str``.

        Typically, pandas will store string data in an object-dtype array.
        When converting values to a dtype, e.g. in ``.astype``, we need to
        be more specific, we need the actual underlying type.

        Returns
        -------
        >>> SparseDtype(int, 1)._subtype_with_str
        dtype('int64')

        >>> SparseDtype(object, 1)._subtype_with_str
        dtype('O')

        >>> dtype = SparseDtype(str, "")
        >>> dtype.subtype
        dtype('O')

        >>> dtype._subtype_with_str
        <class 'str'>
        """
        if isinstance(self.fill_value, str):
            return type(self.fill_value)
        return self.subtype

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: common-type hook. Returns none if any dtype is a non-sparse extension
        # dtype. Otherwise takes the first sparse fill value; when the
        # performance-warnings option is on and the fill values differ (ignoring the
        # all-missing case) emits a performance warning that the first is kept. The
        # result is a SparseDtype whose subtype is the numpy common type of all subtypes
        # and whose fill value is the first one.
        # TODO for now only handle SparseDtypes and numpy dtypes => extend
        # with other compatible extension dtypes
        from pandas.core.dtypes.cast import np_find_common_type

        if any(
            isinstance(x, ExtensionDtype) and not isinstance(x, SparseDtype)
            for x in dtypes
        ):
            return None

        fill_values = [x.fill_value for x in dtypes if isinstance(x, SparseDtype)]
        fill_value = fill_values[0]

        from pandas import isna

        # np.nan isn't a singleton, so we may end up with multiple
        # NaNs here, so we ignore the all NA case too.
        if get_option("performance_warnings") and (
            not (len(set(fill_values)) == 1 or isna(fill_values).all())
        ):
            warnings.warn(
                "Concatenating sparse arrays with multiple fill "
                f"values: '{fill_values}'. Picking the first and "
                "converting the rest.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )
        np_dtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        # error: Argument 1 to "np_find_common_type" has incompatible type
        # "*Generator[Any | dtype[Any] | ExtensionDtype, None, None]";
        # expected "dtype[Any]"  [arg-type]
        return SparseDtype(np_find_common_type(*np_dtypes), fill_value=fill_value)  # type: ignore [arg-type]


@register_extension_dtype
@set_module("pandas")
class ArrowDtype(StorageExtensionDtype):
    # [CLASS DOC]
    # Extension dtype for pyarrow-backed data (experimental). Subclass of the storage
    # extension dtype with storage fixed to the string pyarrow. Parameter:
    # pyarrow_dtype, a pyarrow data type instance. Identity metadata: storage and
    # pyarrow type. Name is the pyarrow type string followed by the storage in brackets.
    # Provides: the Python scalar type matching the arrow type, derived numpy dtype,
    # kind and item size, string parsing, numeric/boolean flags, conversion from arrow
    # and common-type resolution. Paired with the Arrow extension array.
    """
    An ExtensionDtype for PyArrow data types.

    .. warning::

       ArrowDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    While most ``dtype`` arguments can accept the "string"
    constructor, e.g. ``"int64[pyarrow]"``, ArrowDtype is useful
    if the data type contains parameters like ``pyarrow.timestamp``.

    Parameters
    ----------
    pyarrow_dtype : pa.DataType
        An instance of a `pyarrow.DataType <https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions>`__.

    Attributes
    ----------
    pyarrow_dtype

    Methods
    -------
    None

    Returns
    -------
    ArrowDtype

    See Also
    --------
    DataFrame.convert_dtypes : Convert columns to the best possible dtypes.

    Examples
    --------
    >>> import pyarrow as pa
    >>> pd.ArrowDtype(pa.int64())
    int64[pyarrow]

    Types with parameters must be constructed with ArrowDtype.

    >>> pd.ArrowDtype(pa.timestamp("s", tz="America/New_York"))
    timestamp[s, tz=America/New_York][pyarrow]
    >>> pd.ArrowDtype(pa.list_(pa.int64()))
    list<item: int64>[pyarrow]
    """

    _metadata = ("storage", "pyarrow_dtype")  # type: ignore[assignment]

    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        # [FUNCTION DOC]
        # Parameter: pyarrow_dtype. Sets storage to pyarrow via the parent. Raises an
        # import error if pyarrow is not installed, and a value error if the argument is
        # not a pyarrow data type. Stores it as an attribute.
        super().__init__("pyarrow")
        if not HAS_PYARROW:
            raise ImportError(
                f"pyarrow>={PYARROW_MIN_VERSION} is required for ArrowDtype"
            )
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise ValueError(
                f"pyarrow_dtype ({pyarrow_dtype}) must be an instance "
                f"of a pyarrow.DataType. Got {type(pyarrow_dtype)} instead."
            )
        self.pyarrow_dtype = pyarrow_dtype

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the name.
        return self.name

    def __hash__(self) -> int:
        # [FUNCTION DOC]
        # Hash of the string form.
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC]
        # If other is not the same class, defers to the parent comparison (name string
        # match). Otherwise equal when the pyarrow types are equal.
        if not isinstance(other, type(self)):
            return super().__eq__(other)
        return self.pyarrow_dtype == other.pyarrow_dtype

    @property
    def type(self):
        # [FUNCTION DOC]
        # Property. Maps the pyarrow type to the Python scalar class: integer to int,
        # floating to float, string/large string to str, binary types to bytes, boolean
        # to bool, duration to Timedelta for nanosecond units else timedelta, timestamp
        # to Timestamp for nanoseconds else datetime, date to date, time to time,
        # decimal to Decimal, dictionary to the categorical placeholder type, list/large
        # list/fixed-size list/map to list, struct to dict, null to the pyarrow null
        # type's own class, and extension types to the type of their storage. Raises a
        # not-implemented error for anything else.
        """
        Returns associated scalar type.
        """
        pa_type = self.pyarrow_dtype
        if pa.types.is_integer(pa_type):
            return int
        elif pa.types.is_floating(pa_type):
            return float
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return str
        elif (
            pa.types.is_binary(pa_type)
            or pa.types.is_fixed_size_binary(pa_type)
            or pa.types.is_large_binary(pa_type)
        ):
            return bytes
        elif pa.types.is_boolean(pa_type):
            return bool
        elif pa.types.is_duration(pa_type):
            if pa_type.unit == "ns":
                return Timedelta
            else:
                return timedelta
        elif pa.types.is_timestamp(pa_type):
            if pa_type.unit == "ns":
                return Timestamp
            else:
                return datetime
        elif pa.types.is_date(pa_type):
            return date
        elif pa.types.is_time(pa_type):
            return time
        elif pa.types.is_decimal(pa_type):
            return Decimal
        elif pa.types.is_dictionary(pa_type):
            # TODO: Potentially change this & CategoricalDtype.type to
            #  something more representative of the scalar
            return CategoricalDtypeType
        elif pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return list
        elif pa.types.is_fixed_size_list(pa_type):
            return list
        elif pa.types.is_map(pa_type):
            return list
        elif pa.types.is_struct(pa_type):
            return dict
        elif pa.types.is_null(pa_type):
            # TODO: None? pd.NA? pa.null?
            return type(pa_type)
        elif isinstance(pa_type, pa.BaseExtensionType):
            return type(self)(pa_type.storage_type).type
        raise NotImplementedError(pa_type)

    @property
    def name(self) -> str:  # type: ignore[override]
        # [FUNCTION DOC]
        # Property. The pyarrow type text followed by the storage in brackets.
        """
        A string identifying the data type.
        """
        return f"{self.pyarrow_dtype!s}[{self.storage}]"

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        # [FUNCTION DOC]
        # Cached. Returns the numpy dtype for the arrow type: timestamps and durations
        # use datetime64 and timedelta64 with the arrow unit (working around pyarrow
        # always reporting nanoseconds); string types give the numpy unicode string
        # dtype; others use pyarrow's own mapping; if the mapping is unsupported, object
        # dtype.
        """Return an instance of the related numpy dtype"""
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # pa.timestamp(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow timestamp units.
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"datetime64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_duration(self.pyarrow_dtype):
            # pa.duration(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow duration units
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"timedelta64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_string(self.pyarrow_dtype) or pa.types.is_large_string(
            self.pyarrow_dtype
        ):
            # pa.string().to_pandas_dtype() = object which we don't want
            return np.dtype(str)
        try:
            return np.dtype(self.pyarrow_dtype.to_pandas_dtype())
        except (NotImplementedError, TypeError):
            return np.dtype(object)

    @cache_readonly
    def kind(self) -> str:
        # [FUNCTION DOC]
        # Cached. Returns the datetime kind code for timestamp types (mirroring the
        # timezone-aware pandas dtype), otherwise the numpy dtype's kind.
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # To mirror DatetimeTZDtype
            return "M"
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        # [FUNCTION DOC]
        # Cached. Returns the numpy item size for boolean; otherwise the arrow bit width
        # divided by eight, falling back to the numpy item size for variable-width or
        # unsupported types (value, attribute or not-implemented errors).
        """
        Return the number of bytes in this dtype.

        For Arrow-backed dtypes:
        - Returns the fixed-width bit size divided by 8 for standard fixed-width types.
        - For boolean types, returns the NumPy itemsize.
        - Falls back to the NumPy dtype itemsize for variable-width & unsupported types.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> dtype = pd.ArrowDtype(pa.int32())
        >>> dtype.itemsize
        4

        >>> dtype = pd.ArrowDtype(pa.bool_())
        >>> dtype.itemsize  # falls back to numpy dtype
        1
        """
        if pa.types.is_boolean(self.pyarrow_dtype):
            return self.numpy_dtype.itemsize

        # Use pyarrow itemsize for fixed-width data types
        # e.g. int32 -> 32 bits // 8 = 4 bytes
        try:
            return self.pyarrow_dtype.bit_width // 8
        except (ValueError, AttributeError, NotImplementedError):
            return self.numpy_dtype.itemsize

    def construct_array_type(self) -> type_t[ArrowExtensionArray]:
        # [FUNCTION DOC]
        # Returns the ArrowExtensionArray class (imported lazily).
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.arrow import ArrowExtensionArray

        return ArrowExtensionArray

    @classmethod
    def construct_from_string(cls, string: str) -> ArrowDtype:
        # [FUNCTION DOC]
        # Class method. Parameter: string of the form arrow type alias followed by the
        # pyarrow storage tag in brackets. Raises a type error if not a string, if it
        # lacks the storage suffix, or for the string aliases (deliberately excluded so
        # the string dtype takes over); an import error if pyarrow is missing. The alias
        # before the suffix is resolved by pyarrow; if unknown and the text has bracket
        # or parenthesis parameters, a temporal parser is tried, and failure raises a
        # not-implemented error asking for an explicit dtype object; otherwise a type
        # error says the alias is invalid. Returns the dtype.
        """
        Construct this type from a string.

        Parameters
        ----------
        string : str
            string should follow the format f"{pyarrow_type}[pyarrow]"
            e.g. int64[pyarrow]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if not string.endswith("[pyarrow]"):
            raise TypeError(f"'{string}' must end with '[pyarrow]'")
        if string in ("string[pyarrow]", "str[pyarrow]"):
            # Ensure Registry.find skips ArrowDtype to use StringDtype instead
            raise TypeError("string[pyarrow] should be constructed by StringDtype")
        if not HAS_PYARROW:
            raise ImportError(
                f"pyarrow>={PYARROW_MIN_VERSION} is required for ArrowDtype"
            )

        base_type = string[:-9]  # get rid of "[pyarrow]"
        try:
            pa_dtype = pa.type_for_alias(base_type)
        except ValueError as err:
            has_parameters = re.search(r"[\[\(].*[\]\)]", base_type)
            if has_parameters:
                # Fallback to try common temporal types
                try:
                    return cls._parse_temporal_dtype_string(base_type)
                except (NotImplementedError, ValueError):
                    # Fall through to raise with nice exception message below
                    pass

                raise NotImplementedError(
                    "Passing pyarrow type specific parameters "
                    f"({has_parameters.group()}) in the string is not supported. "
                    "Please construct an ArrowDtype object with a pyarrow_dtype "
                    "instance with specific parameters."
                ) from err
            raise TypeError(f"'{base_type}' is not a valid pyarrow data type.") from err
        return cls(pa_dtype)

    # TODO(arrow#33642): This can be removed once supported by pyarrow
    @classmethod
    def _parse_temporal_dtype_string(cls, string: str) -> ArrowDtype:
        # [FUNCTION DOC]
        # HELPER: class method. Parameter: string with the storage suffix already
        # removed and a bracket present. Currently only handles the timestamp form with
        # a unit and a time zone (optionally prefixed by the tz assignment marker),
        # producing a dtype. Raises a value error if the closing bracket is missing, a
        # not-implemented error for other types. Contains an assertion that a comma is
        # present.
        """
        Construct a temporal ArrowDtype from string.
        """
        # we assume
        #  1) "[pyarrow]" has already been stripped from the end of our string.
        #  2) we know "[" is present
        head, tail = string.split("[", 1)

        if not tail.endswith("]"):
            raise ValueError
        tail = tail[:-1]

        if head == "timestamp":
            assert "," in tail  # otherwise type_for_alias should work
            unit, tz = tail.split(",", 1)
            unit = unit.strip()
            tz = tz.strip()
            if tz.startswith("tz="):
                tz = tz[3:]

            pa_type = pa.timestamp(unit, tz=tz)
            dtype = cls(pa_type)
            return dtype

        raise NotImplementedError(string)

    @property
    def _is_numeric(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true for integer, floating and decimal arrow types (not
        # boolean).
        """
        Whether columns with this dtype should be considered numeric.
        """
        # TODO: pa.types.is_boolean?
        return (
            pa.types.is_integer(self.pyarrow_dtype)
            or pa.types.is_floating(self.pyarrow_dtype)
            or pa.types.is_decimal(self.pyarrow_dtype)
        )

    @property
    def _is_boolean(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property; true for the arrow boolean type.
        """
        Whether this dtype should be considered boolean.
        """
        return pa.types.is_boolean(self.pyarrow_dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # [FUNCTION DOC]
        # HELPER: common-type hook. Arrow null-typed dtypes are ignored; the remaining
        # arrow dtypes are replaced by their numpy dtypes and the generic common type is
        # found. If that is not a numpy dtype returns none; otherwise it is converted
        # back to an arrow type and wrapped, returning none if arrow cannot represent
        # it.
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        # Mirrors BaseMaskedDtype
        from pandas.core.dtypes.cast import find_common_type

        null_dtype = type(self)(pa.null())

        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, ArrowDtype) else dtype
                for dtype in dtypes
                if dtype != null_dtype
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            return None
        try:
            pa_dtype = pa.from_numpy_dtype(new_dtype)
            return type(self)(pa_dtype)
        except NotImplementedError:
            return None

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> ArrowExtensionArray:
        # [FUNCTION DOC]
        # Converts a pyarrow array or chunked array to an Arrow extension array of this
        # dtype. Parameter: array. Casts it safely to this dtype's arrow type and wraps
        # it with the array class.
        # KNOWN ISSUE: the docstring says it constructs integer or floating arrays, but
        # the method builds the generic Arrow extension array.
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        array_class = self.construct_array_type()
        arr = array.cast(self.pyarrow_dtype, safe=True)
        return array_class(arr)
