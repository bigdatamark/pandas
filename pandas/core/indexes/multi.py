# [MODULE DOC]
# Implementation of MultiIndex, the hierarchical (multi-level) index, together
# with the compiled-engine adapter classes it uses and a few module-level
# helpers.
# Public contents: the MultiIndex class (re-labelled to the top-level package
# namespace); five engine classes mapping label-code combinations to unsigned
# integers of increasing width (8, 16, 32, 64 bits) or to Python integers when
# those would overflow; a decorator letting methods accept either a name or a
# names keyword; and helper functions: the lexsort depth of code arrays, the
# sparsification of labels for display, the missing-value text for a dtype, level
# dropping after a key lookup, coercion of code arrays to read-only small
# integers, a list-likeness check for the level/codes setters, and the cartesian
# product of arrays (the module's other public function).
# Relationships: MultiIndex subclasses the generic Index. Its state is a list of
# level indexes (the unique labels per level), a parallel list of integer code
# arrays (positions into the levels, minus one for missing) and a list of level
# names, stored as frozen lists (the names as a plain list). Operations lean on the
# frozen list type, the categorical factorization helpers, the sorting utilities
# and the compiled index engines. The data frame, series and categorical index
# classes are imported only for type checking or lazily in methods.
# Import-time side effects: none beyond defining the classes (the class body
# also installs disabled arithmetic operators, see the class documentation).

from __future__ import annotations

from collections.abc import (
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
)
from functools import wraps
from itertools import zip_longest
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    algos as libalgos,
    index as libindex,
    lib,
)
from pandas._libs.hashtable import duplicated
from pandas._typing import (
    AnyAll,
    AnyArrayLike,
    Axis,
    DropKeep,
    DtypeObj,
    F,
    IgnoreRaise,
    IndexLabel,
    IndexT,
    NaPosition,
    Scalar,
    Shape,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    InvalidIndexError,
    PerformanceWarning,
    UnsortedIndexError,
)
from pandas.util._decorators import (
    cache_readonly,
    set_module,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,
    maybe_unbox_numpy_scalar,
)
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
    array_equivalent,
    isna,
)

import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
    Categorical,
    ExtensionArray,
)
from pandas.core.arrays.categorical import (
    factorize_from_iterables,
    recode_for_categories,
)
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    ensure_index,
    get_unanimous_names,
)
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
    get_group_index,
    lexsort_indexer,
)

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas import (
        CategoricalIndex,
        DataFrame,
        Series,
    )


class MultiIndexUInt64Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    # [CLASS DOC]
    # HELPER: lookup engine for a MultiIndex whose label-combination count fits in an
    # unsigned 64-bit integer. Inherits from the compiled multi-index codes engine
    # base and the unsigned 64-bit hash engine. Class attributes: the base engine
    # class and the dtype name for the encoded integer codes. All behavior comes from
    # the bases.
    """Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 64 bits integers.
    """

    _base = libindex.UInt64Engine
    _codes_dtype = "uint64"


class MultiIndexUInt32Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt32Engine):
    # [CLASS DOC]
    # HELPER: same as the 64-bit engine adapter but for combinations that fit in an
    # unsigned 32-bit integer (codes dtype uint32, base engine the 32-bit engine).
    """Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 32 bits integers.
    """

    _base = libindex.UInt32Engine
    _codes_dtype = "uint32"


class MultiIndexUInt16Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt16Engine):
    # [CLASS DOC]
    # HELPER: same as the 64-bit engine adapter but for combinations that fit in an
    # unsigned 16-bit integer (codes dtype uint16, base engine the 16-bit engine).
    """Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 16 bits integers.
    """

    _base = libindex.UInt16Engine
    _codes_dtype = "uint16"


class MultiIndexUInt8Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt8Engine):
    # [CLASS DOC]
    # HELPER: same as the 64-bit engine adapter but for combinations that fit in an
    # unsigned 8-bit integer (codes dtype uint8, base engine the 8-bit engine).
    """Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 8 bits integers.
    """

    _base = libindex.UInt8Engine
    _codes_dtype = "uint8"


class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    # [CLASS DOC]
    # HELPER: engine adapter for the extreme case where the number of label
    # combinations overflows 64 bits: it combines the compiled multi-index codes
    # engine with the object engine and stores Python integers (codes dtype object).
    """Manages a MultiIndex by mapping label combinations to positive integers.

    This class manages those (extreme) cases in which the number of possible
    label combinations overflows the 64 bits integers, and uses an ObjectEngine
    containing Python integers.
    """

    _base = libindex.ObjectEngine
    _codes_dtype = "object"


def names_compat(meth: F) -> F:
    # [FUNCTION DOC]
    # HELPER: decorator so a method accepts either a name keyword or a names keyword,
    # for sharing code with the base class. Parameter meth is the function to wrap.
    # Returns a wrapper (preserving the function's metadata) that raises a type error
    # if both keywords are supplied, renames a name keyword to names, and calls the
    # wrapped function with the remaining arguments. The result is cast to the
    # decorated function's type for type checking.
    """
    A decorator to allow either `name` or `names` keyword but not both.

    This makes it easier to share code with base class.
    """

    @wraps(meth)
    def new_meth(self_or_cls, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: the wrapper produced by the name-compatibility decorator. Takes the
        # instance or class as first argument followed by arbitrary arguments. Raises a
        # type error when both name and names keywords are present; otherwise moves the
        # value of name to names and calls the decorated method, returning its result.
        if "name" in kwargs and "names" in kwargs:
            raise TypeError("Can only provide one of `names` and `name`")
        if "name" in kwargs:
            kwargs["names"] = kwargs.pop("name")

        return meth(self_or_cls, *args, **kwargs)

    return cast(F, new_meth)


@set_module("pandas")
class MultiIndex(Index):
    # [CLASS DOC]
    # Hierarchical index with several levels. Inherits from Index. State: _levels is
    # a frozen list of level indexes holding unique labels; _codes is a frozen list of
    # integer arrays (one per level, equal lengths) giving for every row the position
    # of its label in the level, or minus one for missing; _names is a list with one
    # name per level; sortorder is the number of leading levels known to be
    # lexicographically sorted (or none). Class attributes start as empty containers so
    # a partially built object works. Compared attributes: the names. Invariants (checked
    # when integrity verification is on): equal numbers of levels and code arrays,
    # equal code lengths, codes within level bounds, unique level values, and
    # sortorder not exceeding the true lexsort depth. Construction goes through the
    # constructor or the from_* class methods (arrays, tuples, product, frame).
    # Provides level access and replacement, deletion/swap/reorder of levels,
    # sorting by level, location and slicing lookups, indexer computation, set
    # operations, conversion to tuples or frames, and pickling support. Values are
    # exposed as an object array of tuples. Class-level aliases: rename is the same
    # function as the base class name setter; the duplicated-detection helper used by
    # the base class is the same as the public duplicated method; and every
    # arithmetic, reflected, in-place and unary numeric operator (add, subtract, power,
    # multiply, division variants, modulo, divmod, negation, positive, absolute,
    # invert) is replaced by an invalid-operation stub that raises a type error.
    """
    A multi-level, or hierarchical, index object for pandas objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    name : Label
        Kept for compatibility with 1-dimensional Index. Should not be used.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    Attributes
    ----------
    names
    levels
    codes
    nlevels
    levshape
    dtypes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_indexer
    get_loc
    get_locs
    get_loc_level
    drop

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : The base pandas Index type.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__
    for more.

    Examples
    --------
    A new ``MultiIndex`` is typically constructed using one of the helper
    methods :meth:`MultiIndex.from_arrays`, :meth:`MultiIndex.from_product`
    and :meth:`MultiIndex.from_tuples`. For example (using ``.from_arrays``):

    >>> arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
    >>> pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               names=['number', 'color'])

    See further examples for how to construct a MultiIndex in the doc strings
    of the mentioned helper methods.
    """

    _hidden_attrs = Index._hidden_attrs | frozenset()

    # initialize to zero-length tuples to make everything work
    _typ = "multiindex"
    _names: list[Hashable | None] = []
    _levels = FrozenList()
    _codes = FrozenList()
    _comparables = ["names"]

    sortorder: int | None

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        copy: bool = False,
        name=None,
        verify_integrity: bool = True,
    ) -> Self:
        # [FUNCTION DOC]
        # Construct a MultiIndex from explicit levels and codes. Parameters: levels is a
        # sequence of arrays (unique labels per level); codes a same-length sequence of
        # integer arrays; sortorder an optional int (level of lexicographic sortedness);
        # names an optional sequence of level names; copy (default false) copies the
        # meta-data; name is an alias for names kept for Index compatibility (if given it
        # overrides names); verify_integrity (default true) validates the structure and
        # rewrites codes that point to missing level entries to minus one.
        # Raises a type error if levels or codes is missing, a value error if their
        # lengths differ or are zero, and (with verification) a value error from the
        # integrity check. The object is built without calling a base constructor: an
        # empty cache is created, levels and codes are set without validation, names are
        # set (validated), sortorder is converted to an int, the identity token reset, and
        # the reference tracker set to none. Returns the new index.
        # compat with Index
        if name is not None:
            names = name
        if levels is None or codes is None:
            raise TypeError("Must pass both levels and codes")
        if len(levels) != len(codes):
            raise ValueError("Length of levels and codes must be the same.")
        if len(levels) == 0:
            raise ValueError("Must pass non-zero number of levels/codes")

        result = object.__new__(cls)
        result._cache = {}

        # we've already validated levels and codes, so shortcut here
        result._set_levels(levels, copy=copy, validate=False)
        result._set_codes(codes, copy=copy, validate=False)

        result._names = [None] * len(levels)
        if names is not None:
            # handles name validation
            result._set_names(names)

        if sortorder is not None:
            result.sortorder = int(sortorder)
        else:
            result.sortorder = sortorder

        if verify_integrity:
            new_codes = result._verify_integrity()
            result._codes = new_codes

        result._reset_identity()
        result._references = None

        return result

    def _validate_codes(self, level: Index, code: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: remaps codes that point to missing entries in a level to minus one.
        # Parameters: level is the level's labels and code is the integer code array.
        # If the level contains any missing values (not-a-number, not-a-time or none),
        # codes pointing to them are replaced by minus one. Returns the (possibly new)
        # code array.
        # KNOWN ISSUE: the docstring lists the types of the two parameters the wrong way round: it says code is an Index and level is a numpy array, but the code treats level as the labels and code as the numpy array.
        """
        Reassign code values as -1 if their corresponding levels are NaN.

        Parameters
        ----------
        code : Index
            Code to reassign.
        level : np.ndarray
            Level to check for missing values (NaN, NaT, None).

        Returns
        -------
        new code where code value = -1 if it corresponds
        to a level with missing values (NaN, NaT, None).
        """
        null_mask = isna(level)
        if np.any(null_mask):
            code = np.where(null_mask[code], -1, code)
        return code

    def _verify_integrity(
        self,
        codes: list | None = None,
        levels: list | None = None,
        levels_to_verify: list[int] | range | None = None,
    ) -> FrozenList:
        # [FUNCTION DOC]
        # HELPER: validates the internal consistency of codes and levels. Parameters:
        # codes (default none: use the current codes), levels (default none: use the
        # current levels) and levels_to_verify (default none: all levels; a list or range
        # of positions). Note that empty lists count as not given, because of the
        # truthiness test.
        # KNOWN ISSUE: the docstring names the last parameter levels_to_validate, but the parameter is called levels_to_verify.
        # Raises a value error when the number of levels and codes differ, when code
        # lengths differ, when a code is at or above the level length, when a code is
        # below minus one, when a verified level is not unique, or when sortorder exceeds
        # the actual lexsort depth. After the checks it returns a frozen list of codes in
        # which the verified levels have codes pointing at missing labels set to minus
        # one while others are untouched. Does not check that the cached number of levels
        # or sortorder match.
        """
        Parameters
        ----------
        codes : optional list
            Codes to check for validity. Defaults to current codes.
        levels : optional list
            Levels to check for validity. Defaults to current levels.
        levels_to_validate: optional list
            Specifies the levels to verify.

        Raises
        ------
        ValueError
            If length of levels and codes don't match, if the codes for any
            level would exceed level bounds, or there are any duplicate levels.

        Returns
        -------
        new codes where code value = -1 if it corresponds to a
        NaN level.
        """
        # NOTE: Currently does not check, among other things, that cached
        # nlevels matches nor that sortorder matches actually sortorder.
        codes = codes or self.codes
        levels = levels or self.levels
        if levels_to_verify is None:
            levels_to_verify = range(len(levels))

        if len(levels) != len(codes):
            raise ValueError(
                "Length of levels and codes must match. NOTE: "
                "this index is in an inconsistent state."
            )
        codes_length = len(codes[0])
        for i in levels_to_verify:
            level = levels[i]
            level_codes = codes[i]

            if len(level_codes) != codes_length:
                raise ValueError(
                    f"Unequal code lengths: {[len(code_) for code_ in codes]}"
                )
            if len(level_codes) and level_codes.max() >= len(level):
                raise ValueError(
                    f"On level {i}, code max ({level_codes.max()}) >= length of "
                    f"level ({len(level)}). NOTE: this index is in an "
                    "inconsistent state"
                )
            if len(level_codes) and level_codes.min() < -1:
                raise ValueError(f"On level {i}, code value ({level_codes.min()}) < -1")
            if not level.is_unique:
                raise ValueError(
                    f"Level values must be unique: {list(level)} on level {i}"
                )
        if self.sortorder is not None:
            if self.sortorder > _lexsort_depth(self.codes, self.nlevels):
                raise ValueError(
                    "Value for sortorder must be inferior or equal to actual "
                    f"lexsort_depth: sortorder {self.sortorder} "
                    f"with lexsort_depth {_lexsort_depth(self.codes, self.nlevels)}"
                )

        result_codes = []
        for i in range(len(levels)):
            if i in levels_to_verify:
                result_codes.append(self._validate_codes(levels[i], codes[i]))
            else:
                result_codes.append(codes[i])

        new_codes = FrozenList(result_codes)
        return new_codes

    @classmethod
    def from_arrays(
        cls,
        arrays,
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | lib.NoDefault = lib.no_default,
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Class method converting a sequence of array-likes (one per level) to a
        # MultiIndex. Parameters: arrays; sortorder (default none); names (default the
        # no-default sentinel meaning take each array's name attribute, or none). Raises
        # a type error if arrays is not list-like or any element is not list-like, and a
        # value error if lengths differ. Iterators are first consumed into a list. Each
        # array is factorized into codes and levels and the instance is built without
        # integrity verification. Returns the new index.
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
        >>> pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        error_msg = "Input must be a list / sequence of array-likes."
        if not is_list_like(arrays):
            raise TypeError(error_msg)
        if is_iterator(arrays):
            arrays = list(arrays)

        # Check if elements of array are list-like
        for array in arrays:
            if not is_list_like(array):
                raise TypeError(error_msg)

        # Check if lengths of all arrays are equal or not,
        # raise ValueError, if not
        for i in range(1, len(arrays)):
            if len(arrays[i]) != len(arrays[i - 1]):
                raise ValueError("all arrays must be same length")

        codes, levels = factorize_from_iterables(arrays)
        if names is lib.no_default:
            names = [getattr(arr, "name", None) for arr in arrays]

        return cls(
            levels=levels,
            codes=codes,
            sortorder=sortorder,
            names=names,
            verify_integrity=False,
        )

    @classmethod
    @names_compat
    def from_tuples(
        cls,
        tuples: Iterable[tuple[Hashable, ...]],
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | None = None,
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Class method converting tuples to a MultiIndex (decorated so it accepts name or
        # names). Parameters: tuples is a list-like of tuple-likes (an iterator is
        # consumed, an index or array is converted to object values); sortorder
        # (default none); names (default none). Raises a type error if tuples is not
        # list-like, or is empty without names. Special cases: if every element is an
        # empty tuple, a single level with a zero code for each is built; empty input with
        # names creates empty arrays, one per name; lists, arrays and indexes are
        # transposed with fast helpers; other iterables are zipped with missing entries
        # padded by not-a-number. The result is then built with the arrays constructor.
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, "red"), (1, "blue"), (2, "red"), (2, "blue")]
        >>> pd.MultiIndex.from_tuples(tuples, names=("number", "color"))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        if not is_list_like(tuples):
            raise TypeError("Input must be a list / sequence of tuple-likes.")
        if is_iterator(tuples):
            tuples = list(tuples)
        tuples = cast(Collection[tuple[Hashable, ...]], tuples)

        # handling the empty tuple cases
        if len(tuples) and all(isinstance(e, tuple) and not e for e in tuples):
            codes = [np.zeros(len(tuples))]
            levels = [Index(com.asarray_tuplesafe(tuples, dtype=np.dtype("object")))]
            return cls(
                levels=levels,
                codes=codes,
                sortorder=sortorder,
                names=names,
                verify_integrity=False,
            )

        arrays: list[Sequence[Hashable]]
        if len(tuples) == 0:
            if names is None:
                raise TypeError("Cannot infer number of levels from empty list")
            # error: Argument 1 to "len" has incompatible type "Hashable";
            # expected "Sized"
            arrays = [[]] * len(names)  # type: ignore[arg-type]
        elif isinstance(tuples, (np.ndarray, Index)):
            if isinstance(tuples, Index):
                tuples = np.asarray(tuples._values)

            arrays = list(lib.tuples_to_object_array(tuples).T)
        elif isinstance(tuples, list):
            arrays = list(lib.to_object_array_tuples(tuples).T)
        else:
            arrs = zip_longest(*tuples, fillvalue=np.nan)
            arrays = cast(list[Sequence[Hashable]], arrs)

        return cls.from_arrays(arrays, sortorder=sortorder, names=names)

    @classmethod
    def from_product(
        cls,
        iterables: Sequence[Iterable[Hashable]],
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | lib.NoDefault = lib.no_default,
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Class method building the cartesian product of several iterables as a
        # MultiIndex. Parameters: iterables (list-like of iterables, an iterator is
        # consumed); sortorder (default none); names (default the no-default sentinel
        # meaning use each iterable's name attribute if any). Raises a type error if
        # iterables is not list-like. Each iterable is factorized into codes and levels,
        # the codes are expanded into their cartesian product with the module helper,
        # and the index is built with integrity verification at its default. Returns the
        # new index.
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.
            If not explicitly provided, names will be inferred from the
            elements of iterables if an element has a name attribute.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ["green", "purple"]
        >>> pd.MultiIndex.from_product([numbers, colors], names=["number", "color"])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """

        if not is_list_like(iterables):
            raise TypeError("Input must be a list / sequence of iterables.")
        if is_iterator(iterables):
            iterables = list(iterables)

        codes, levels = factorize_from_iterables(iterables)
        if names is lib.no_default:
            names = [getattr(it, "name", None) for it in iterables]

        # codes are all ndarrays, so cartesian_product is lossless
        codes = cartesian_product(codes)
        return cls(levels, codes, sortorder=sortorder, names=names)

    @classmethod
    def from_frame(
        cls,
        df: DataFrame,
        sortorder: int | None = None,
        names: Sequence[Hashable] | Hashable | None = None,
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Class method building a MultiIndex from the columns of a data frame.
        # Parameters: df must be a data frame (a type error is raised otherwise);
        # sortorder (default none); names (default none: the column names are used, or
        # the column label tuples when the columns are themselves a MultiIndex; a given
        # sequence replaces them). Each column becomes one level through the arrays
        # constructor. Returns the new index.
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        sortorder : int, optional
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [["HI", "Temp"], ["HI", "Precip"], ["NJ", "Temp"], ["NJ", "Precip"]],
        ...     columns=["a", "b"],
        ... )
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> pd.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> pd.MultiIndex.from_frame(df, names=["state", "observation"])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        if not isinstance(df, ABCDataFrame):
            raise TypeError("Input must be a DataFrame")

        column_names, columns = zip(*df.items(), strict=True)
        names = column_names if names is None else names
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    # --------------------------------------------------------------------

    @cache_readonly
    def _values(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: cached object array of tuples, one per row; overrides the base class
        # because this class has no single data array. For each level the labels are
        # looked up by codes (a missing code yields the level's missing value);
        # categorical levels are first expanded to their underlying values, and
        # extension or datetime-like levels converted to object arrays. The per-level
        # arrays are zipped into a single object array of tuples by a fast helper.
        # We override here, since our parent uses _data, which we don't use.
        values = []

        for i in range(self.nlevels):
            index = self.levels[i]
            codes = self.codes[i]

            vals = index
            if isinstance(vals.dtype, CategoricalDtype):
                vals = cast("CategoricalIndex", vals)
                vals = vals._data._internal_get_values()

            if isinstance(vals.dtype, ExtensionDtype) or lib.is_np_dtype(
                vals.dtype, "mM"
            ):
                vals = vals.astype(object)

            array_vals = np.asarray(vals)
            array_vals = algos.take_nd(array_vals, codes, fill_value=index._na_value)
            values.append(array_vals)

        arr = lib.fast_zip(values)
        return arr

    @property
    def values(self) -> np.ndarray:
        # [FUNCTION DOC]
        # Property: the values as an object array of tuples (the cached result).
        return self._values

    @property
    def array(self):
        # [FUNCTION DOC]
        # Property: always raises a value error because a MultiIndex has no single backing
        # array; the message suggests converting to a numpy array of tuples instead.
        """
        Raises a ValueError for `MultiIndex` because there's no single
        array backing a MultiIndex.

        Raises
        ------
        ValueError
        """
        raise ValueError(
            "MultiIndex has no single backing array. Use "
            "'MultiIndex.to_numpy()' to get a NumPy array of tuples."
        )

    @cache_readonly
    def dtypes(self) -> Series:
        # [FUNCTION DOC]
        # Cached property: a series of the dtype of each level, indexed by the level
        # names (missing names are filled with positional placeholder names).
        """
        Return the dtypes as a Series for the underlying MultiIndex.

        See Also
        --------
        Index.dtype : Return the dtype object of the underlying data.
        Series.dtypes : Return the data type of the underlying Series.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product(
        ...     [(0, 1, 2), ("green", "purple")], names=["number", "color"]
        ... )
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.dtypes
        number     int64
        color     object
        dtype: object
        """
        from pandas import Series

        names = com.fill_missing_names(self.names)
        return Series([level.dtype for level in self.levels], index=Index(names))

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Return the number of rows, taken from the length of the first code array.
        return len(self.codes[0])

    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property: the number of rows (the length). Overrides the base class to avoid
        # materializing the tuple values.
        """
        Return the number of elements in the underlying data.
        """
        # override Index.size to avoid materializing _values
        return len(self)

    # --------------------------------------------------------------------
    # Levels Methods

    @cache_readonly
    def levels(self) -> FrozenList:
        # [FUNCTION DOC]
        # Cached property: the levels as a frozen list of Index objects, each renamed to
        # its level name and flagged so that setting a name on a level directly is
        # refused. Cached so the index engine is not rebuilt on repeated access
        # (GH 31648).
        # KNOWN ISSUE: the docstring says the attribute returns a tuple of Index objects, but it returns a frozen list.
        """
        Levels of the MultiIndex.

        Levels refer to the different hierarchical levels or layers in a MultiIndex.
        In a MultiIndex, each level represents a distinct dimension or category of
        the index.

        To access the levels, you can use the levels attribute of the MultiIndex,
        which returns a tuple of Index objects. Each Index object represents a
        level in the MultiIndex and contains the unique values found in that
        specific level.

        If a MultiIndex is created with levels A, B, C, and the DataFrame using
        it filters out all rows of the level C, MultiIndex.levels will still
        return A, B, C.

        See Also
        --------
        MultiIndex.codes : The codes of the levels in the MultiIndex.
        MultiIndex.get_level_values : Return vector of label values for requested
            level.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [["mammal"], ("goat", "human", "cat", "dog")],
        ...     names=["Category", "Animals"],
        ... )
        >>> leg_num = pd.DataFrame(data=(4, 2, 4, 4), index=index, columns=["Legs"])
        >>> leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 human       2
                 cat         4
                 dog         4

        >>> leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])

        MultiIndex levels will not change even if the DataFrame using the MultiIndex
        does not contain all them anymore.
        See how "human" is not in the DataFrame, but it is still in levels:

        >>> large_leg_num = leg_num[leg_num.Legs > 2]
        >>> large_leg_num
                          Legs
        Category Animals
        mammal   goat        4
                 cat         4
                 dog         4

        >>> large_leg_num.index.levels
        FrozenList([['mammal'], ['cat', 'dog', 'goat', 'human']])
        """
        # Use cache_readonly to ensure that self.get_locs doesn't repeatedly
        # create new IndexEngine
        # https://github.com/pandas-dev/pandas/issues/31648
        result = [
            x._rename(name=name)
            for x, name in zip(self._levels, self._names, strict=True)
        ]
        for level in result:
            # disallow midx.levels[0].name = "foo"
            level._no_setting_name = True
        return FrozenList(result)

    def _set_levels(
        self,
        levels,
        *,
        level=None,
        copy: bool = False,
        validate: bool = True,
        verify_integrity: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: sets the levels in place (not part of the public property because users
        # must not set levels directly). Keyword-only parameters: levels (sequence of
        # index-likes), level (positions or names of levels to replace; default none for
        # all), copy (default false), validate (default true) and verify_integrity
        # (default false). With validation: raises a value error for zero levels, for a
        # length mismatch with the number of levels (when replacing all), or with the
        # number of selected levels. Replacement levels are converted to indexes and
        # views taken. With verify_integrity the codes are checked and rewritten for the
        # replaced levels. Then the new levels are stored, the names are re-applied if
        # any exists, and the cache is cleared. Returns nothing.
        # This is NOT part of the levels property because it should be
        # externally not allowed to set levels. User beware if you change
        # _levels directly
        if validate:
            if len(levels) == 0:
                raise ValueError("Must set non-zero number of levels.")
            if level is None and len(levels) != self.nlevels:
                raise ValueError("Length of levels must match number of levels.")
            if level is not None and len(levels) != len(level):
                raise ValueError("Length of levels must match length of level.")

        if level is None:
            new_levels = FrozenList(
                ensure_index(lev, copy=copy)._view() for lev in levels
            )
            level_numbers: range | list[int] = range(len(new_levels))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_levels_list = list(self._levels)
            for lev_num, lev in zip(level_numbers, levels, strict=True):
                new_levels_list[lev_num] = ensure_index(lev, copy=copy)._view()
            new_levels = FrozenList(new_levels_list)

        if verify_integrity:
            new_codes = self._verify_integrity(
                levels=new_levels, levels_to_verify=level_numbers
            )
            self._codes = new_codes

        names = self.names
        self._levels = new_levels
        if any(names):
            self._set_names(names)

        self._reset_cache()

    def set_levels(
        self, levels, *, level=None, verify_integrity: bool = True
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Return a new MultiIndex with levels replaced. Parameters: levels (sequence or
        # sequence of sequences); level (default none: all; otherwise int, name, or a
        # list of them, same length as levels); verify_integrity (keyword-only, default
        # true: checks new levels are consistent with the codes). A single index-like
        # or array-like input is wrapped as a list. Helper converts scalar level and
        # levels to lists. Works on a view of this index with its identity reset, so the
        # original is unchanged. Raises a value error (from the setter) on mismatched
        # lengths or a failed integrity check. Returns the new index.
        """
        Set new levels on MultiIndex. Defaults to returning new index.

        The `set_levels` method provides a flexible way to change the levels of a
        `MultiIndex`. This is particularly useful when you need to update the
        index structure of your DataFrame without altering the data. The method
        returns a new `MultiIndex` unless the operation is performed in-place,
        ensuring that the original index remains unchanged unless explicitly
        modified.

        The method checks the integrity of the new levels against the existing
        codes by default, but this can be disabled if you are confident that
        your levels are consistent with the underlying data. This can be useful
        when you want to perform optimizations or make specific adjustments to
        the index levels that do not strictly adhere to the original structure.

        Parameters
        ----------
        levels : sequence or list of sequence
            New level(s) to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        MultiIndex
            A new `MultiIndex` with the updated levels.

        See Also
        --------
        MultiIndex.set_codes : Set new codes on the existing `MultiIndex`.
        MultiIndex.remove_unused_levels : Create new MultiIndex from current that
            removes unused levels.
        Index.set_names : Set Index or MultiIndex name.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, "one"),
        ...         (1, "two"),
        ...         (2, "one"),
        ...         (2, "two"),
        ...         (3, "one"),
        ...         (3, "two"),
        ...     ],
        ...     names=["foo", "bar"],
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two'),
            (3, 'one'),
            (3, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_levels([["a", "b", "c"], [1, 2]])
        MultiIndex([('a', 1),
                    ('a', 2),
                    ('b', 1),
                    ('b', 2),
                    ('c', 1),
                    ('c', 2)],
                   names=['foo', 'bar'])
        >>> idx.set_levels(["a", "b", "c"], level=0)
        MultiIndex([('a', 'one'),
                    ('a', 'two'),
                    ('b', 'one'),
                    ('b', 'two'),
                    ('c', 'one'),
                    ('c', 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_levels(["a", "b"], level="bar")
        MultiIndex([(1, 'a'),
                    (1, 'b'),
                    (2, 'a'),
                    (2, 'b'),
                    (3, 'a'),
                    (3, 'b')],
                   names=['foo', 'bar'])

        If any of the levels passed to ``set_levels()`` exceeds the
        existing length, all of the values from that argument will
        be stored in the MultiIndex levels, though the values will
        be truncated in the MultiIndex output.

        >>> idx.set_levels([["a", "b", "c"], [1, 2, 3, 4]], level=[0, 1])
        MultiIndex([('a', 1),
            ('a', 2),
            ('b', 1),
            ('b', 2),
            ('c', 1),
            ('c', 2)],
           names=['foo', 'bar'])
        >>> idx.set_levels([["a", "b", "c"], [1, 2, 3, 4]], level=[0, 1]).levels
        FrozenList([['a', 'b', 'c'], [1, 2, 3, 4]])
        """

        if isinstance(levels, Index):
            pass
        elif is_array_like(levels):
            levels = Index(levels)
        elif is_list_like(levels):
            levels = list(levels)

        level, levels = _require_listlike(level, levels, "Levels")
        idx = self._view()
        idx._reset_identity()
        idx._set_levels(
            levels, level=level, validate=True, verify_integrity=verify_integrity
        )
        return idx

    @property
    def nlevels(self) -> int:
        # [FUNCTION DOC]
        # Property: the number of levels.
        """
        Integer number of levels in this MultiIndex.

        See Also
        --------
        MultiIndex.levels : Get the levels of the MultiIndex.
        MultiIndex.codes : Get the codes of the MultiIndex.
        MultiIndex.from_arrays : Convert arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.nlevels
        3
        """
        return len(self._levels)

    @property
    def levshape(self) -> Shape:
        # [FUNCTION DOC]
        # Property: a tuple of the lengths of each level (the count of unique labels per
        # level).
        """
        A tuple representing the length of each level in the MultiIndex.

        In a `MultiIndex`, each level can contain multiple unique values. The
        `levshape` property provides a quick way to assess the size of each
        level by returning a tuple where each entry represents the number of
        unique values in that specific level. This is particularly useful in
        scenarios where you need to understand the structure and distribution
        of your index levels, such as when working with multidimensional data.

        See Also
        --------
        MultiIndex.shape : Return a tuple of the shape of the MultiIndex.
        MultiIndex.levels : Returns the levels of the MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.levshape
        (1, 1, 1)
        """
        return tuple(len(x) for x in self.levels)

    # --------------------------------------------------------------------
    # Codes Methods

    @property
    def codes(self) -> FrozenList:
        # [FUNCTION DOC]
        # Property: the codes as a frozen list of integer arrays, one per level.
        # KNOWN ISSUE: the docstring Returns section says a tuple of numpy arrays, but the property returns a frozen list.
        """
        Codes of the MultiIndex.

        Codes are the position of the index value in the list of level values
        for each level.

        Returns
        -------
        tuple of numpy.ndarray
            The codes of the MultiIndex. Each array in the tuple corresponds
            to a level in the MultiIndex.

        See Also
        --------
        MultiIndex.set_codes : Set new codes on MultiIndex.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
        >>> mi = pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
        >>> mi.codes
        FrozenList([[0, 0, 1, 1], [1, 0, 1, 0]])
        """
        return self._codes

    def _set_codes(
        self,
        codes,
        *,
        level=None,
        copy: bool = False,
        validate: bool = True,
        verify_integrity: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: sets the codes in place. Keyword-only parameters: codes (sequence of
        # integer sequences), level (levels to replace; default none for all), copy
        # (default false), validate (default true), verify_integrity (default false).
        # With validation: raises a value error if the number of code arrays does not
        # match the number of levels or the number of selected levels. Codes are coerced
        # to the smallest suitable integer dtype for their level size and made
        # read-only views. With verify_integrity the replaced levels are checked and
        # rewritten. The new codes are stored and the cache cleared. Returns nothing.
        if validate:
            if level is None and len(codes) != self.nlevels:
                raise ValueError("Length of codes must match number of levels")
            if level is not None and len(codes) != len(level):
                raise ValueError("Length of codes must match length of levels.")

        level_numbers: list[int] | range
        if level is None:
            new_codes = FrozenList(
                _coerce_indexer_frozen(level_codes, lev, copy=copy).view()
                for lev, level_codes in zip(self._levels, codes, strict=True)
            )
            level_numbers = range(len(new_codes))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_codes_list = list(self._codes)
            for lev_num, level_codes in zip(level_numbers, codes, strict=True):
                lev = self.levels[lev_num]
                new_codes_list[lev_num] = _coerce_indexer_frozen(
                    level_codes, lev, copy=copy
                )
            new_codes = FrozenList(new_codes_list)

        if verify_integrity:
            new_codes = self._verify_integrity(
                codes=new_codes, levels_to_verify=level_numbers
            )

        self._codes = new_codes

        self._reset_cache()

    def set_codes(
        self, codes, *, level=None, verify_integrity: bool = True
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Return a new MultiIndex with codes replaced. Parameters: codes (sequence or
        # sequence of sequences); level (default none: all; else int, name or list);
        # verify_integrity (keyword-only, default true). Operates on a view of this
        # index with its identity reset, so the original is unchanged. Raises a value
        # error on mismatched lengths or a failed integrity check. Returns the new index.
        # KNOWN ISSUE: the docstring Returns section mentions None if inplace is true, but there is no inplace parameter and the method always returns a new index.
        """
        Set new codes on MultiIndex. Defaults to returning new index.

        Parameters
        ----------
        codes : sequence or list of sequence
            New codes to apply.
        level : int, level name, or sequence of int/level names (default None)
            Level(s) to set (None for all levels).
        verify_integrity : bool, default True
            If True, checks that levels and codes are compatible.

        Returns
        -------
        new index (of same type and class...etc) or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        MultiIndex.set_levels : Set new levels on MultiIndex.
        MultiIndex.codes : Get the codes of the levels in the MultiIndex.
        MultiIndex.levels : Get the levels of the MultiIndex.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [(1, "one"), (1, "two"), (2, "one"), (2, "two")], names=["foo", "bar"]
        ... )
        >>> idx
        MultiIndex([(1, 'one'),
            (1, 'two'),
            (2, 'one'),
            (2, 'two')],
           names=['foo', 'bar'])

        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([1, 0, 1, 0], level=0)
        MultiIndex([(2, 'one'),
                    (1, 'two'),
                    (2, 'one'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([0, 0, 1, 1], level="bar")
        MultiIndex([(1, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (2, 'two')],
                   names=['foo', 'bar'])
        >>> idx.set_codes([[1, 0, 1, 0], [0, 0, 1, 1]], level=[0, 1])
        MultiIndex([(2, 'one'),
                    (1, 'one'),
                    (2, 'two'),
                    (1, 'two')],
                   names=['foo', 'bar'])
        """

        level, codes = _require_listlike(level, codes, "Codes")
        idx = self._view()
        idx._reset_identity()
        idx._set_codes(codes, level=level, verify_integrity=verify_integrity)
        return idx

    # --------------------------------------------------------------------
    # Index Internals

    @cache_readonly
    def _engine(self):
        # [FUNCTION DOC]
        # HELPER: cached lookup engine. Computes for each level the number of bits needed
        # for its labels (log base two of the level size plus the shift reserved for
        # missing values), accumulates them from the right to get bit offsets so the
        # combined integer sorts like the codes, and picks the smallest engine class
        # whose integer width fits the total bits: 8, 16, 32, 64 bits, or Python
        # integers beyond 64. Returns the engine instantiated with levels, codes and
        # offsets.
        # KNOWN ISSUE: the inline comments for the 32-bit and 16-bit thresholds say to use an 8-bit or 16-bit unsigned integer in the wrong place (the code returns the 32-bit engine for more than 16 bits, but the comment says 8-bit); the returned engines are right, the comments are misleading.
        # Calculate the number of bits needed to represent labels in each
        # level, as log2 of their sizes:
        # NaN values are shifted to 1 and missing values in other while
        # calculating the indexer are shifted to 0
        sizes = np.ceil(
            np.log2(
                [len(level) + libindex.multiindex_nulls_shift for level in self.levels]
            )
        )

        # Sum bit counts, starting from the _right_....
        lev_bits = np.cumsum(sizes[::-1])[::-1]

        # ... in order to obtain offsets such that sorting the combination of
        # shifted codes (one for each level, resulting in a unique integer) is
        # equivalent to sorting lexicographically the codes themselves. Notice
        # that each level needs to be shifted by the number of bits needed to
        # represent the _previous_ ones:
        offsets = np.concatenate([lev_bits[1:], [0]])
        # Downcast the type if possible, to prevent upcasting when shifting codes:
        offsets = offsets.astype(np.min_scalar_type(int(offsets[0])))

        # Check the total number of bits needed for our representation:
        if lev_bits[0] > 64:
            # The levels would overflow a 64 bit uint - use Python integers:
            return MultiIndexPyIntEngine(self.levels, self.codes, offsets)
        if lev_bits[0] > 32:
            # The levels would overflow a 32 bit uint - use uint64
            return MultiIndexUInt64Engine(self.levels, self.codes, offsets)
        if lev_bits[0] > 16:
            # The levels would overflow a 16 bit uint - use uint8
            return MultiIndexUInt32Engine(self.levels, self.codes, offsets)
        if lev_bits[0] > 8:
            # The levels would overflow a 8 bit uint - use uint16
            return MultiIndexUInt16Engine(self.levels, self.codes, offsets)
        # The levels fit in an 8 bit uint - use uint8
        return MultiIndexUInt8Engine(self.levels, self.codes, offsets)

    # Return type "Callable[..., MultiIndex]" of "_constructor" incompatible with return
    # type "Type[MultiIndex]" in supertype "Index"
    @property
    def _constructor(self) -> Callable[..., MultiIndex]:  # type: ignore[override]
        # [FUNCTION DOC]
        # HELPER: property returning the callable used to build new indexes from tuples
        # (the from-tuples class method of this class). The signature differs from the
        # supertype's on purpose.
        return type(self).from_tuples

    def _shallow_copy(self, values: np.ndarray, name=lib.no_default) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: creates a new MultiIndex from an array of tuples without sharing other
        # state. Parameters: values (array of tuples) and name (default no-default
        # sentinel meaning keep the current names; otherwise the new names). Returns
        # the result of the from-tuples constructor with no sortorder.
        """
        Create a new Index with the same class as the caller, don't copy the
        data, use the same object attributes with passed in attributes taking
        precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new Index, optional
        name : Label, defaults to self.name
        """
        names = name if name is not lib.no_default else self.names

        return type(self).from_tuples(values, sortorder=None, names=names)

    def _view(self) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: creates a lightweight view: a new MultiIndex with the same levels,
        # codes, sortorder and names, built without integrity verification, with a copy of
        # the cache except the cached levels entry (GH32669). Returns it.
        result = type(self)(
            levels=self.levels,
            codes=self.codes,
            sortorder=self.sortorder,
            names=self.names,
            verify_integrity=False,
        )
        result._cache = self._cache.copy()
        result._reset_cache("levels")  # GH32669
        return result

    # --------------------------------------------------------------------

    # error: Signature of "copy" incompatible with supertype "Index"
    def copy(  # type: ignore[override]
        self,
        names=None,
        deep: bool = False,
        name=None,
    ) -> Self:
        # [FUNCTION DOC]
        # Return a copy of the index. Parameters: names (default none; sequence of level
        # names to set on the copy), deep (default false) and name (compat alias for
        # names; default none). Names are validated together (error if both are given)
        # with the deep flag deciding whether the names are copied. With deep set, the
        # levels and codes are deep-copied; otherwise they are shared. A new index of the
        # same class is built with the same sortorder, without integrity verification;
        # the cache is copied except the cached levels entry (GH32669). A shallow copy
        # keeps the original's identity token; a deep copy gets a fresh one. Returns the
        # new index.
        # KNOWN ISSUE: the summary line of the docstring claims that dtype, levels and codes can be passed and set on the copy, but only names (and deep) are accepted.
        """
        Make a copy of this object. Names, dtype, levels and codes can be passed and \
        will be set on new copy.

        The `copy` method provides a mechanism to create a duplicate of an
        existing MultiIndex object. This is particularly useful in scenarios where
        modifications are required on an index, but the original MultiIndex should
        remain unchanged. By specifying the `deep` parameter, users can control
        whether the copy should be a deep or shallow copy, providing flexibility
        depending on the size and complexity of the MultiIndex.

        Parameters
        ----------
        names : sequence, optional
            Names to set on the new MultiIndex object.
        deep : bool, default False
            If False, the new object will be a shallow copy. If True, a deep copy
            will be attempted. Deep copying can be potentially expensive for large
            MultiIndex objects.
        name : Label
            Kept for compatibility with 1-dimensional Index. Should not be used.

        Returns
        -------
        MultiIndex
            A new MultiIndex object with the specified modifications.

        See Also
        --------
        MultiIndex.from_arrays : Convert arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Convert DataFrame to MultiIndex.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )
        >>> mi.copy()
        MultiIndex([('a', 'b', 'c')],
                   )
        """
        names = self._validate_names(name=name, names=names, deep=deep)
        keep_id = not deep
        levels, codes = None, None

        if deep:
            from copy import deepcopy

            levels = deepcopy(self.levels)
            codes = deepcopy(self.codes)

        levels = levels if levels is not None else self.levels
        codes = codes if codes is not None else self.codes

        new_index = type(self)(
            levels=levels,
            codes=codes,
            sortorder=self.sortorder,
            names=names,
            verify_integrity=False,
        )
        new_index._cache = self._cache.copy()
        new_index._reset_cache("levels")  # GH32669
        if keep_id:
            new_index._id = self._id
        return new_index

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        # [FUNCTION DOC]
        # Array interface. Parameters: dtype (default none) and copy (default none).
        # Raises a value error if copy is explicitly false, because a copy cannot be
        # avoided when building the array. If copy is true a fresh array is made from
        # the values (with the dtype), since the cached values array must not be shared.
        # Otherwise returns the cached values array itself (the dtype argument is
        # ignored in this case). Returns an object array of tuples.
        """the array interface, return my values"""
        if copy is False:
            # self.values is always a newly construct array, so raise.
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        if copy is True:
            # explicit np.array call to ensure a copy is made and unique objects
            # are returned, because self.values is cached
            return np.array(self.values, dtype=dtype)
        return self.values

    def view(self, cls=None) -> Self:
        # [FUNCTION DOC]
        # Return a view: a copy that shares this index's identity token. Parameter cls is
        # accepted for compatibility and ignored. Returns the new index.
        """this is defined as a copy with the same identity"""
        result = self.copy()
        result._id = self._id
        return result

    def __contains__(self, key: Any) -> bool:
        # [FUNCTION DOC]
        # Membership test. Parameter key must be hashable (the hash call raises a type
        # error otherwise). Returns true if location lookup succeeds; any lookup error
        # (key, index, type or value error) gives false.
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
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi
        MultiIndex([('a', 'b', 'c')],
                   )

        >>> "a" in mi
        True
        >>> "x" in mi
        False
        """
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError, ValueError):
            return False

    @cache_readonly
    def dtype(self) -> np.dtype:
        # [FUNCTION DOC]
        # Cached property: the dtype, always the object dtype.
        return np.dtype("O")

    @cache_readonly
    def _is_memory_usage_qualified(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached flag saying whether the info display must qualify the memory
        # usage as a lower bound: true if any level has an object dtype or a Python-
        # backed string dtype.
        """return a boolean if we need a qualified .info display"""

        def f(dtype) -> bool:
            # [FUNCTION DOC]
            # HELPER: predicate for one dtype: true when it is the object dtype or a string
            # dtype whose storage is the Python one.
            return is_object_dtype(dtype) or (
                is_string_dtype(dtype) and dtype.storage == "python"
            )

        return any(f(level.dtype) for level in self.levels)

    # Cannot determine type of "memory_usage"
    def memory_usage(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # Return the memory usage in bytes. Parameter deep (default false) asks for deep
        # inspection of object data. Avoids materializing the tuple values by using the
        # private byte counter. Returns an integer.
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
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"], ["c"]])
        >>> mi.memory_usage()
        81
        """
        # we are overwriting our base class to avoid
        # computing .values here which could materialize
        # a tuple representation unnecessarily
        return self._nbytes(deep)

    @cache_readonly
    def nbytes(self) -> int:
        # [FUNCTION DOC]
        # Cached property: the number of bytes of the underlying data (shallow count).
        """return the number of bytes in the underlying data"""
        return self._nbytes(False)

    def _nbytes(self, deep: bool = False) -> int:
        # [FUNCTION DOC]
        # HELPER: computes the number of bytes. Parameter deep (default false). Sum of
        # the levels' memory usage (deep or not), the bytes of the code arrays, and the
        # sizes of the level name objects (a fixed 24 bytes when the size cannot be
        # determined), plus the engine's size if the engine is already cached.
        """
        return the number of bytes in the underlying data
        deeply introspect the level data if deep=True

        include the engine hashtable

        *this is in internal routine*

        """
        # for implementations with no useful getsizeof (PyPy)
        objsize = 24

        level_nbytes = sum(i.memory_usage(deep=deep) for i in self.levels)
        label_nbytes = sum(i.nbytes for i in self.codes)
        names_nbytes = sum(getsizeof(i, objsize) for i in self.names)
        result = level_nbytes + label_nbytes + names_nbytes

        # include our engine hashtable, only if it's already cached
        if "_engine" in self._cache:
            result += self._engine.sizeof(deep=deep)
        return result

    # --------------------------------------------------------------------
    # Rendering Methods

    def _formatter_func(self, tup):
        # [FUNCTION DOC]
        # HELPER: formats one row tuple. Parameter tup is a tuple with one element per
        # level. Applies each level's own formatter function to the corresponding
        # element and returns a tuple of the results.
        """
        Formats each item in tup according to its level's formatter function.
        """
        formatter_funcs = (level._formatter_func for level in self.levels)
        return tuple(func(val) for func, val in zip(formatter_funcs, tup, strict=True))

    def _get_values_for_csv(
        self, *, na_rep: str = "nan", **kwargs
    ) -> npt.NDArray[np.object_]:
        # [FUNCTION DOC]
        # HELPER: formats values for CSV output. Keyword-only parameters: na_rep (default
        # the text nan) and extra formatting keywords passed to the levels. Each level is
        # formatted by its own method; for levels with missing codes an extra entry with
        # the missing representation is appended (levels turned to strings) and the
        # codes pointing at missing become that entry (on a writable copy; an assertion
        # requires the original codes to be read-only). A single-level index is
        # formatted through a plain Index built by taking the formatted level by codes.
        # Otherwise a temporary MultiIndex of the formatted levels is built and its values
        # returned. Returns an object array.
        new_levels = []
        new_codes = []

        # go through the levels and format them
        for level, level_codes in zip(self.levels, self.codes, strict=True):
            level_strs = level._get_values_for_csv(na_rep=na_rep, **kwargs)
            # add nan values, if there are any
            mask = level_codes == -1
            if mask.any():
                nan_index = len(level_strs)
                # numpy 1.21 deprecated implicit string casting
                level_strs = level_strs.astype(str)
                level_strs = np.append(level_strs, na_rep)
                assert not level_codes.flags.writeable  # i.e. copy is needed
                level_codes = level_codes.copy()  # make writeable
                level_codes[mask] = nan_index
            new_levels.append(level_strs)
            new_codes.append(level_codes)

        if len(new_levels) == 1:
            # a single-level multi-index
            return Index(
                new_levels[0].take(new_codes[0]), copy=False
            )._get_values_for_csv()
        else:
            # reconstruct the multi-index
            mi = MultiIndex(
                levels=new_levels,
                codes=new_codes,
                names=self.names,
                sortorder=self.sortorder,
                verify_integrity=False,
            )
            return mi._values

    def _format_multi(
        self,
        *,
        include_names: bool,
        sparsify: bool | None | lib.NoDefault,
        formatter: Callable | None = None,
    ) -> list:
        # [FUNCTION DOC]
        # HELPER: builds the string columns for displaying the index. Keyword-only
        # parameters: include_names (whether to add each level name as the first entry of
        # its column), sparsify (true, false, none or the no-default sentinel; none reads
        # the display option for multi-level sparsification) and formatter (optional
        # callable). Returns an empty list for an empty index. Each level is formatted by
        # taking its labels by codes and applying the flat formatter, with missing codes
        # shown with a type-specific missing text; a level with no labels (all-missing)
        # is formatted element by element. Names are prepended (escaped), and when
        # sparsify is on, repeated leading labels are blanked using the sparsification
        # helper (the sentinel is the empty string, or the no-default marker when that was
        # given; a falsy-but-not-bool value is rejected by an assertion). Returns the list
        # of columns, each a list of strings.
        if len(self) == 0:
            return []

        stringified_levels = []
        for lev, level_codes in zip(self.levels, self.codes, strict=True):
            na = _get_na_rep(lev.dtype)

            if len(lev) > 0:
                taken = formatted = lev.take(level_codes)
                formatted = taken._format_flat(include_name=False, formatter=formatter)

                # we have some NA
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()

            else:
                # weird all NA case
                formatted = [
                    pprint_thing(na if isna(x) else x, escape_chars=("\t", "\r", "\n"))
                    for x in algos.take_nd(lev._values, level_codes)
                ]
            stringified_levels.append(formatted)

        result_levels = []
        for lev, lev_name in zip(stringified_levels, self.names, strict=True):
            level = []

            if include_names:
                level.append(
                    pprint_thing(lev_name, escape_chars=("\t", "\r", "\n"))
                    if lev_name is not None
                    else ""
                )

            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)

        if sparsify is None:
            sparsify = get_option("display.multi_sparse")

        if sparsify:
            sentinel: Literal[""] | bool | lib.NoDefault = ""
            # GH3547 use value of sparsify as sentinel if it's "Falsey"
            assert isinstance(sparsify, bool) or sparsify is lib.no_default
            if sparsify is lib.no_default:
                sentinel = sparsify
            # little bit of a kludge job for #1217
            result_levels = sparsify_labels(
                result_levels, start=int(include_names), sentinel=sentinel
            )

        return result_levels

    # --------------------------------------------------------------------
    # Names Methods

    def _get_names(self) -> FrozenList:
        # [FUNCTION DOC]
        # HELPER: getter of the names property: returns the level names as a frozen list.
        return FrozenList(self._names)

    def _set_names(self, names, *, level=None) -> None:
        # [FUNCTION DOC]
        # HELPER: setter of the names property (also called by the constructor).
        # Parameters: names (a list-like of hashable names; a non-list-like such as a
        # single string raises a value error, GH 15110) and level (default none: all
        # levels; otherwise level numbers or names, converted via the level lookup).
        # Raises a value error if the count of names does not match the count of levels
        # or selected levels, and a type error if any non-none name is not hashable
        # (GH 20527). The names are stored in place (mutation) and the cached levels are
        # discarded since their names are stale. Returns nothing.
        # KNOWN ISSUE: the docstring documents the first parameter as values, but the parameter is named names.
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None

        Raises
        ------
        TypeError if each name is not hashable.

        Notes
        -----
        sets names on levels. WARNING: mutates!

        Note that you generally want to set this *after* changing levels, so
        that it only acts on copies
        """
        # GH 15110
        # Don't allow a single string for names in a MultiIndex
        if names is not None and not is_list_like(names):
            raise ValueError("Names should be list-like for a MultiIndex")
        names = list(names)

        if level is not None and len(names) != len(level):
            raise ValueError("Length of names must match length of level.")
        if level is None and len(names) != self.nlevels:
            raise ValueError(
                "Length of names must match number of levels in MultiIndex."
            )

        if level is None:
            level = range(self.nlevels)
        else:
            level = (self._get_level_number(lev) for lev in level)

        # set the name
        for lev, name in zip(level, names, strict=True):
            if name is not None:
                # GH 20527
                # All items in 'names' need to be hashable:
                if not is_hashable(name):
                    raise TypeError(
                        f"{type(self).__name__}.name must be a hashable type"
                    )
            self._names[lev] = name

        # If .levels has been accessed, the .name of each level in our cache
        # will be stale.
        self._reset_cache("levels")

    names = property(
        fset=_set_names,
        fget=_get_names,
        doc="""
        Names of levels in MultiIndex.

        This attribute provides access to the names of the levels in a `MultiIndex`.
        The names are stored as a `FrozenList`, which is an immutable list-like
        container. Each name corresponds to a level in the `MultiIndex`, and can be
        used to identify or manipulate the levels individually.

        See Also
        --------
        MultiIndex.set_names : Set Index or MultiIndex name.
        MultiIndex.rename : Rename specific levels in a MultiIndex.
        Index.names : Get names on index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ...     [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z']
        ... )
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])
        >>> mi.names
        FrozenList(['x', 'y', 'z'])
        """,
    )

    # --------------------------------------------------------------------

    @cache_readonly
    def inferred_type(self) -> str:
        # [FUNCTION DOC]
        # Cached property: constant string "mixed".
        return "mixed"

    def _get_level_number(self, level) -> int:
        # [FUNCTION DOC]
        # HELPER: converts a level name or number to a zero-based level number.
        # Parameter level is a name or an integer. A name that occurs more than once
        # raises a value error unless the argument is an integer. An exact match in the
        # names wins even for integers; otherwise a non-integer raises a key error
        # saying the level was not found; a negative integer counts from the end and
        # raises an index error if still negative; an integer beyond the last level
        # raises an index error. Returns the level number.
        count = self.names.count(level)
        if (count > 1) and not is_integer(level):
            raise ValueError(
                f"The name {level} occurs multiple times, use a level number"
            )
        try:
            level = self.names.index(level)
        except ValueError as err:
            if not is_integer(level):
                raise KeyError(f"Level {level} not found") from err
            if level < 0:
                level += self.nlevels
                if level < 0:
                    orig_level = level - self.nlevels
                    raise IndexError(
                        f"Too many levels: Index has only {self.nlevels} levels, "
                        f"{orig_level} is not a valid level number"
                    ) from err
            # Note: levels are zero-based
            elif level >= self.nlevels:
                raise IndexError(
                    f"Too many levels: Index has only {self.nlevels} levels, "
                    f"not {level + 1}"
                ) from err
        return level

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: true when the rows are in non-decreasing lexicographic order.
        # False if any code is minus one (missing). If every level is monotonic
        # increasing, the sortedness of the code arrays is checked directly (GH27495).
        # Otherwise a lexicographic sort order is computed from the level values
        # (least significant key first) and checked to be increasing; on a type error
        # (mixed types) the tuple values are checked instead.
        """
        Return a boolean if the values are equal or increasing.
        """
        if any(-1 in code for code in self.codes):
            return False

        if all(level.is_monotonic_increasing for level in self.levels):
            # If each level is sorted, we can operate on the codes directly. GH27495
            return libalgos.is_lexsorted(
                [x.astype("int64", copy=False) for x in self.codes]
            )

        # reversed() because lexsort() wants the most significant key last.
        values = [
            self._get_level_values(i)._values for i in reversed(range(len(self.levels)))
        ]
        try:
            # error: Argument 1 to "lexsort" has incompatible type
            # "List[Union[ExtensionArray, ndarray[Any, Any]]]";
            # expected "Union[_SupportsArray[dtype[Any]],
            # _NestedSequence[_SupportsArray[dtype[Any]]], bool,
            # int, float, complex, str, bytes, _NestedSequence[Union
            # [bool, int, float, complex, str, bytes]]]"
            sort_order = np.lexsort(values)  # type: ignore[arg-type]
            return Index(sort_order, copy=False).is_monotonic_increasing
        except TypeError:
            # we have mixed types and np.lexsort is not happy
            return Index(self._values, copy=False).is_monotonic_increasing

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        # [FUNCTION DOC]
        # Cached property: true when the rows are in non-increasing order, computed as
        # whether the reversed index is monotonic increasing.
        """
        Return a boolean if the values are equal or decreasing.
        """
        # monotonic decreasing if and only if reverse is monotonic increasing
        return self[::-1].is_monotonic_increasing

    def duplicated(self, keep: DropKeep = "first") -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Mark duplicated rows. Parameter keep: first (default; mark all but the first
        # occurrence), last, or false (mark all duplicates). Combines the codes into one
        # group id per row without sorting or dropping nulls and delegates to the
        # compiled duplicate finder. Returns a boolean array. The private duplicated
        # attribute is aliased to this method.
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        np.ndarray[bool]
            A numpy array of boolean values indicating duplicate index values.

        See Also
        --------
        Series.duplicated : Equivalent method on pandas.Series.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> mi = pd.MultiIndex.from_arrays((list("abca"), list("defd")))
        >>> mi.duplicated()
        array([False, False, False,  True])

        which is equivalent to

        >>> mi.duplicated(keep="first")
        array([False, False, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> mi.duplicated(keep="last")
        array([ True, False, False, False])

        By setting keep on ``False``, all duplicates are True:

        >>> mi.duplicated(keep=False)
        array([ True, False, False,  True])
        """
        shape = tuple(len(lev) for lev in self.levels)
        ids = get_group_index(self.codes, shape, sort=False, xnull=False)

        return duplicated(ids, keep)

    # error: Cannot override final attribute "_duplicated"
    # (previously declared in base class "IndexOpsMixin")
    _duplicated = duplicated  # type: ignore[misc]

    def fillna(self, value):
        # [FUNCTION DOC]
        # Always raises a not-implemented error: filling missing values is not defined for
        # a MultiIndex. Parameter value is ignored.
        """
        fillna is not implemented for MultiIndex
        """
        raise NotImplementedError("fillna is not defined for MultiIndex")

    def dropna(self, how: AnyAll = "any") -> MultiIndex:
        # [FUNCTION DOC]
        # Remove rows with missing entries. Parameter how: any (default; drop rows where
        # any level is missing) or all (drop rows where every level is missing).
        # Raises a value error for any other value. Missing means a code of minus one.
        # Returns a new MultiIndex built by setting the filtered codes on a view.
        """
        Return MultiIndex without NA/NaN values.

        Parameters
        ----------
        how : {'any', 'all'}, default 'any'
            Drop the value when any or all levels are NaN.

        Returns
        -------
        Index
            Returns an MultiIndex object after removing NA/NaN values.

        See Also
        --------
        Index.fillna : Fill NA/NaN values with the specified value.
        Index.isna : Detect missing values.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(([np.nan, np.nan, 2.0], [3.0, np.nan, 4.0]))
        >>> mi.dropna()
        MultiIndex([(2.0, 4.0)],
                   )
        >>> mi.dropna(how="all")
        MultiIndex([(nan, 3.0),
                    (2.0, 4.0)],
                   )
        """
        nans = [level_codes == -1 for level_codes in self.codes]
        if how == "any":
            indexer = np.any(nans, axis=0)
        elif how == "all":
            indexer = np.all(nans, axis=0)
        else:
            raise ValueError(f"invalid how option: {how}")

        new_codes = [level_codes[~indexer] for level_codes in self.codes]
        return self.set_codes(codes=new_codes)

    def _get_level_values(self, level: int, unique: bool = False) -> Index:
        # [FUNCTION DOC]
        # HELPER: returns the labels of one level for every row. Parameters: level (an
        # integer level number) and unique (default false: only distinct codes are
        # used). Takes the level's values by codes (missing becomes the level's missing
        # value) and wraps them in an index of the same kind as the level, with the
        # level's name. Returns that Index.
        """
        Return vector of label values for requested level,
        equal to the length of the index

        **this is an internal method**

        Parameters
        ----------
        level : int
        unique : bool, default False
            if True, drop duplicated values

        Returns
        -------
        Index
        """
        lev = self.levels[level]
        level_codes = self.codes[level]
        name = self._names[level]
        if unique:
            level_codes = algos.unique(level_codes)
        filled = algos.take_nd(lev._values, level_codes, fill_value=lev._na_value)
        return lev._shallow_copy(filled, name=name)

    def get_level_values(self, level) -> Index:
        # [FUNCTION DOC]
        # Return the labels of one level, repeated for every row. Parameter level is a
        # level number or name (resolved with the level lookup; the lookup errors
        # propagate). Returns an Index (of the level's type, possibly cast to a type that
        # can hold missing values). The index has the level's name.
        """
        Return vector of label values for requested level.

        Length of returned vector is equal to the length of the index.
        The `get_level_values` method is a crucial utility for extracting
        specific level values from a `MultiIndex`. This function is particularly
        useful when working with multi-level data, allowing you to isolate
        and manipulate individual levels without having to deal with the
        complexity of the entire `MultiIndex` structure. It seamlessly handles
        both integer and string-based level access, providing flexibility in
        how you can interact with the data. Additionally, this method ensures
        that the returned `Index` maintains the integrity of the original data,
        even when missing values are present, by appropriately casting the
        result to a suitable data type.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        Index
            Values is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        See Also
        --------
        MultiIndex : A multi-level, or hierarchical, index object for pandas objects.
        Index : Immutable sequence used for indexing and alignment.
        MultiIndex.remove_unused_levels : Create new MultiIndex from current that
            removes unused levels.

        Notes
        -----
        If the level contains missing values, the result may be casted to
        ``float`` with missing values specified as ``NaN``. This is because
        the level is converted to a regular ``Index``.

        Examples
        --------
        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list("abc"), list("def")))
        >>> mi.names = ["level_1", "level_2"]

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='str', name='level_1')
        >>> mi.get_level_values("level_2")
        Index(['d', 'e', 'f'], dtype='str', name='level_2')

        If a level contains missing values, the return type of the level
        may be cast to ``float``.

        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).dtypes
        level_0    int64
        level_1    int64
        dtype: object
        >>> pd.MultiIndex.from_arrays([[1, None, 2], [3, 4, 5]]).get_level_values(0)
        Index([1.0, nan, 2.0], dtype='float64')
        """
        level = self._get_level_number(level)
        values = self._get_level_values(level)
        return values

    def unique(self, level=None):
        # [FUNCTION DOC]
        # Return distinct values in order of appearance (not sorted). Parameter level
        # (default none). Without level it removes duplicate rows and returns a
        # MultiIndex; with a level (number or name) it returns an Index of that level's
        # distinct values.
        """
        Return unique values in the index.

        Unique values are returned in order of appearance, this does NOT sort.

        Parameters
        ----------
        level : int or hashable, optional
            Only return values from specified level (for MultiIndex).
            If int, gets the level by integer position, else by level name.

        Returns
        -------
        MultiIndex
            Unique values in the MultiIndex.

        See Also
        --------
        unique : Numpy array of unique values in that column.
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays((list("abca"), list("defd")))
        >>> mi
        MultiIndex([('a', 'd'),
                    ('b', 'e'),
                    ('c', 'f'),
                    ('a', 'd')],
                   )
        >>> mi.unique()
        MultiIndex([('a', 'd'),
                    ('b', 'e'),
                    ('c', 'f')],
                   )
        """
        if level is None:
            return self.drop_duplicates()
        else:
            level = self._get_level_number(level)
            return self._get_level_values(level=level, unique=True)

    def to_frame(
        self,
        index: bool = True,
        name=lib.no_default,
        allow_duplicates: bool = False,
    ) -> DataFrame:
        # [FUNCTION DOC]
        # Convert to a data frame with one column per level. Parameters: index (default
        # true: use this MultiIndex as the frame's index; otherwise a default index);
        # name (default no-default sentinel: use the level names; otherwise a list-like
        # of column names with the same length as the number of levels); allow_duplicates
        # (default false: duplicate column names raise). Raises a type error when name is
        # not list-like, a value error when its length differs from the number of levels,
        # and a value error for duplicate column names without allow_duplicates.
        # Columns are the level value arrays by row, in level order.
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of str, optional
            The passed names should substitute index level names.

        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created.

        Returns
        -------
        DataFrame
            DataFrame representation of the MultiIndex, with levels as columns.

        See Also
        --------
        DataFrame : Two-dimensional, size-mutable, potentially heterogeneous
            tabular data.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a", "b"], ["c", "d"]])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=["x", "y"])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        from pandas import DataFrame

        if name is not lib.no_default:
            if not is_list_like(name):
                raise TypeError("'name' must be a list / sequence of column names.")

            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have same length as number of levels on index."
                )
            idx_names = name
        else:
            idx_names = self._get_level_names()

        if not allow_duplicates and len(set(idx_names)) != len(idx_names):
            raise ValueError(
                "Cannot create duplicate column labels if allow_duplicates is False"
            )

        # Guarantee resulting column order - PY36+ dict maintains insertion order
        result = DataFrame(
            {level: self._get_level_values(level) for level in range(len(self.levels))},
            copy=False,
        )
        result.columns = idx_names

        if index:
            result.index = self
        return result

    # error: Return type "Index" of "to_flat_index" incompatible with return type
    # "MultiIndex" in supertype "Index"
    def to_flat_index(self) -> Index:  # type: ignore[override]
        # [FUNCTION DOC]
        # Convert to a flat Index whose elements are the row tuples (the tuple values,
        # without splitting tuples into levels and without copying). Returns the Index.
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        Returns
        -------
        pd.Index
            Index with the MultiIndex data represented in Tuples.

        See Also
        --------
        MultiIndex.from_tuples : Convert flat index back to MultiIndex.

        Notes
        -----
        This method will simply return the caller if called by anything other
        than a MultiIndex.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [["foo", "bar"], ["baz", "qux"]], names=["a", "b"]
        ... )
        >>> index.to_flat_index()
        Index([('foo', 'baz'), ('foo', 'qux'),
               ('bar', 'baz'), ('bar', 'qux')],
              dtype='object')
        """
        return Index(self._values, tupleize_cols=False, copy=False)

    def _is_lexsorted(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true when the codes are lexicographically sorted over all
        # levels, that is, when the lexsort depth equals the number of levels.
        """
        Return True if the codes are lexicographically sorted.

        Returns
        -------
        bool

        Examples
        --------
        In the below examples, the first level of the MultiIndex is sorted because
        a<b<c, so there is no need to look at the next level.

        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "b", "c"], ["d", "e", "f"]]
        ... )._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "b", "c"], ["d", "f", "e"]]
        ... )._is_lexsorted()
        True

        In case there is a tie, the lexicographical sorting looks
        at the next level of the MultiIndex.

        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ["a", "b", "c"]])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ["a", "c", "b"]])._is_lexsorted()
        False
        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "a", "b", "b"], ["aa", "bb", "aa", "bb"]]
        ... )._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays(
        ...     [["a", "a", "b", "b"], ["bb", "aa", "aa", "bb"]]
        ... )._is_lexsorted()
        False
        """
        return self._lexsort_depth == self.nlevels

    @cache_readonly
    def _lexsort_depth(self) -> int:
        # [FUNCTION DOC]
        # HELPER: cached number of leading levels that are lexicographically sorted. Uses
        # sortorder when it is set; otherwise computes it from the codes with the
        # module helper. Returns an integer.
        """
        Compute and return the lexsort_depth, the number of levels of the
        MultiIndex that are sorted lexically

        Returns
        -------
        int
        """
        if self.sortorder is not None:
            return self.sortorder
        return _lexsort_depth(self.codes, self.nlevels)

    def _sort_levels_monotonic(self, raise_if_incomparable: bool = False) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: returns a MultiIndex whose levels are each sorted (monotonic increasing)
        # while the rows are unchanged in appearance and equal to the original.
        # Parameter raise_if_incomparable (default false): if true, a level that cannot
        # be sorted raises a type error; otherwise such a level is left as is. If the
        # index is already lexsorted and monotonic, returns self. Otherwise each unsorted
        # level is reordered and its codes remapped through the reverse permutation
        # (missing stays minus one). The new index keeps names and sortorder and skips
        # integrity verification.
        """
        This is an *internal* function.

        Create a new MultiIndex from the current to monotonically sorted
        items IN the levels. This does not actually make the entire MultiIndex
        monotonic, JUST the levels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex(
        ...     levels=[["a", "b"], ["bb", "aa"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
        ... )
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )

        >>> mi.sort_values()
        MultiIndex([('a', 'aa'),
                    ('a', 'bb'),
                    ('b', 'aa'),
                    ('b', 'bb')],
                   )
        """
        if self._is_lexsorted() and self.is_monotonic_increasing:
            return self

        new_levels = []
        new_codes = []

        for lev, level_codes in zip(self.levels, self.codes, strict=True):
            if not lev.is_monotonic_increasing:
                try:
                    # indexer to reorder the levels
                    indexer = lev.argsort()
                except TypeError:
                    if raise_if_incomparable:
                        raise
                else:
                    lev = lev.take(indexer)

                    # indexer to reorder the level codes
                    indexer = ensure_platform_int(indexer)
                    ri = lib.get_reverse_indexer(indexer, len(indexer))
                    level_codes = algos.take_nd(ri, level_codes, fill_value=-1)

            new_levels.append(lev)
            new_codes.append(level_codes)

        return MultiIndex(
            new_levels,
            new_codes,
            names=self.names,
            sortorder=self.sortorder,
            verify_integrity=False,
        )

    def remove_unused_levels(self) -> MultiIndex:
        # [FUNCTION DOC]
        # Return a new MultiIndex without level entries that no row uses. The result has
        # the same values and ordering and is equal to the original. For each level, the
        # set of used codes is found by counting (cheaper than unique); if some entries
        # are unused, the used ones are kept in their order of first appearance and the
        # codes remapped (missing stays minus one). If nothing changed a view of the
        # original is returned; otherwise the view has a fresh identity and new levels
        # and codes set without validation.
        # KNOWN ISSUE: when a level has missing labels and the number of used codes equals its length, the loop is terminated early with a break instead of moving on to the next level; the levels after it are then never examined or added, so if an earlier level was changed the new levels list is shorter than the number of levels.
        """
        Create new MultiIndex from current that removes unused levels.

        Unused level(s) means levels that are not expressed in the
        labels. The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will
        also be .equals() to the original.

        The `remove_unused_levels` method is useful in cases where you have a
        MultiIndex with hierarchical levels, but some of these levels are no
        longer needed due to filtering or subsetting operations. By removing
        the unused levels, the resulting MultiIndex becomes more compact and
        efficient, which can improve performance in subsequent operations.

        Returns
        -------
        MultiIndex
            A new MultiIndex with unused levels removed.

        See Also
        --------
        MultiIndex.droplevel : Remove specified levels from a MultiIndex.
        MultiIndex.reorder_levels : Rearrange levels of a MultiIndex.
        MultiIndex.set_levels : Set new levels on a MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_product([range(2), list("ab")])
        >>> mi
        MultiIndex([(0, 'a'),
                    (0, 'b'),
                    (1, 'a'),
                    (1, 'b')],
                   )

        >>> mi[2:]
        MultiIndex([(1, 'a'),
                    (1, 'b')],
                   )

        The 0 from the first level is not represented
        and can be removed

        >>> mi2 = mi[2:].remove_unused_levels()
        >>> mi2.levels
        FrozenList([[1], ['a', 'b']])
        """
        new_levels = []
        new_codes = []

        changed = False
        for lev, level_codes in zip(self.levels, self.codes, strict=True):
            # Since few levels are typically unused, bincount() is more
            # efficient than unique() - however it only accepts positive values
            # (and drops order):
            uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
            has_na = int(len(uniques) and (uniques[0] == -1))

            if len(uniques) != len(lev) + has_na:
                if lev.isna().any() and len(uniques) == len(lev):
                    break
                # We have unused levels
                changed = True

                # Recalculate uniques, now preserving order.
                # Can easily be cythonized by exploiting the already existing
                # "uniques" and stop parsing "level_codes" when all items
                # are found:
                uniques = algos.unique(level_codes)
                if has_na:
                    na_idx = np.where(uniques == -1)[0]
                    # Just ensure that -1 is in first position:
                    uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]

                # codes get mapped from uniques to 0:len(uniques)
                # -1 (if present) is mapped to last position
                code_mapping = np.zeros(len(lev) + has_na)
                # ... and reassigned value -1:
                code_mapping[uniques] = np.arange(len(uniques)) - has_na

                level_codes = code_mapping[level_codes]

                # new levels are simple
                lev = lev.take(uniques[has_na:])

            new_levels.append(lev)
            new_codes.append(level_codes)

        result = self.view()

        if changed:
            result._reset_identity()
            result._set_levels(new_levels, validate=False)
            result._set_codes(new_codes, validate=False)

        return result

    # --------------------------------------------------------------------
    # Pickling Methods

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns the shared base-index reconstruction function with the
        # class and a dictionary of the levels (as a plain list), codes (as a plain
        # list), sortorder and names (as a plain list), and no extra state.
        """Necessary for making this object picklable"""
        d = {
            "levels": list(self.levels),
            "codes": list(self.codes),
            "sortorder": self.sortorder,
            "names": list(self.names),
        }
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------

    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Item access. Parameter key: scalar position, slice, boolean mask, integer array
        # or Index. A zero-dimensional array is unpacked. A scalar (checked by the cast
        # scalar indexer, which rejects floats) returns a tuple with the label of every
        # level at that row, using not-a-number where the code is minus one; an
        # out-of-bounds position raises an index error from the code arrays. Anything
        # else returns a new MultiIndex made by indexing every code array with the key,
        # sharing the levels and names without verification. The result keeps sortorder
        # only for boolean masks and for slices with a positive or missing step;
        # otherwise it is none.
        key = lib.item_from_zerodim(key)
        if is_scalar(key):
            key = com.cast_scalar_indexer(key)

            retval = []
            for lev, level_codes in zip(self.levels, self.codes, strict=True):
                if level_codes[key] == -1:
                    retval.append(np.nan)
                else:
                    retval.append(lev[level_codes[key]])

            return tuple(retval)
        else:
            # in general cannot be sure whether the result will be sorted
            sortorder = None
            if com.is_bool_indexer(key):
                key = np.asarray(key, dtype=bool)
                sortorder = self.sortorder
            elif isinstance(key, slice):
                if key.step is None or key.step > 0:
                    sortorder = self.sortorder
            elif isinstance(key, Index):
                key = np.asarray(key)

            new_codes = [level_codes[key] for level_codes in self.codes]

            return MultiIndex(
                levels=self.levels,
                codes=new_codes,
                names=self.names,
                sortorder=sortorder,
                verify_integrity=False,
            )

    def _getitem_slice(self: MultiIndex, slobj: slice) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: slice fast path. Parameter slobj is a slice. Slices every code array
        # and returns a new MultiIndex of the same class sharing levels and names, keeping
        # sortorder only for a positive or missing step.
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        sortorder = None
        if slobj.step is None or slobj.step > 0:
            sortorder = self.sortorder

        new_codes = [level_codes[slobj] for level_codes in self.codes]

        return type(self)(
            levels=self.levels,
            codes=new_codes,
            names=self._names,
            sortorder=sortorder,
            verify_integrity=False,
        )

    def take(
        self: MultiIndex,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Return a new MultiIndex of the rows at the given positions. Parameters:
        # indices (array-like of integers, converted to platform integers); axis (default
        # 0, not otherwise used); allow_fill (default true: minus one entries mean missing
        # rather than counting from the end) and fill_value (default none: filling is only
        # allowed when a fill value is given); extra keyword arguments are validated for
        # numpy compatibility. Raises a value error (from the fill check) if filling is
        # enabled and negative indices other than minus one appear. If the indices are
        # exactly the full range a copy is returned. Otherwise every code array is taken,
        # and if filling, positions where the index is minus one get the missing code
        # minus one. The result shares levels and names, has no sortorder and no
        # verification. The actual fill value is not used beyond switching filling on.
        # KNOWN ISSUE: the docstring describes the False setting of allow_fill as the default and the True setting as the other one, but the signature default is true.
        """
        Return a new MultiIndex of the values selected by the indices.

        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : {0 or 'index'}, optional
            The axis over which to select values, always 0 or 'index'.
        allow_fill : bool, default True
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
            from the right (the default). This is similar to
            :func:`numpy.take`.

            * True: negative values in `indices` indicate
            missing values. These values are set to `fill_value`. Any other
            other negative values raise a ``ValueError``.

        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays([["a", "b", "c"], [1, 2, 3]])
        >>> idx
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('c', 3)],
                   )
        >>> idx.take([2, 2, 1, 0])
        MultiIndex([('c', 3),
                    ('c', 3),
                    ('b', 2),
                    ('a', 1)],
                   )
        """
        nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)

        # only fill if we are passing a non-None fill_value
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)

        if indices.ndim == 1 and lib.is_range_indexer(indices, len(self)):
            return self.copy()

        na_value = -1

        taken = [lab.take(indices) for lab in self.codes]
        if allow_fill:
            mask = indices == -1
            if mask.any():
                masked = []
                for new_label in taken:
                    label_values = new_label
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked

        return MultiIndex(
            levels=self.levels, codes=taken, names=self.names, verify_integrity=False
        )

    def append(self, other):
        # [FUNCTION DOC]
        # Append other indexes. Parameter other is an index or a list or tuple of
        # indexes. If every element is a MultiIndex with at least as many levels as this
        # one, the result is built from the levels: each level becomes the union of the
        # corresponding levels, each index's codes are recoded into it and concatenated,
        # and a level name is kept only if all indexes agree on it (else none); the result
        # is a MultiIndex without integrity verification and without sortorder.
        # Otherwise the tuple values of all are concatenated and converted with the
        # from-tuples constructor (names dropped); if that raises a type or index error
        # a flat Index of the tuples is returned instead.
        """
        Append a collection of Index options together.

        The `append` method is used to combine multiple `Index` objects into a single
        `Index`. This is particularly useful when dealing with multi-level indexing
        (MultiIndex) where you might need to concatenate different levels of indices.
        The method handles the alignment of the levels and codes of the indices being
        appended to ensure consistency in the resulting `MultiIndex`.

        Parameters
        ----------
        other : Index or list/tuple of indices
            Index or list/tuple of Index objects to be appended.

        Returns
        -------
        Index
            The combined index.

        See Also
        --------
        MultiIndex: A multi-level, or hierarchical, index object for pandas objects.
        Index.append : Append a collection of Index options together.
        concat : Concatenate pandas objects along a particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a"], ["b"]])
        >>> mi
        MultiIndex([('a', 'b')],
                   )
        >>> mi.append(mi)
        MultiIndex([('a', 'b'), ('a', 'b')],
                   )
        """
        if not isinstance(other, (list, tuple)):
            other = [other]

        if all(
            (isinstance(o, MultiIndex) and o.nlevels >= self.nlevels) for o in other
        ):
            codes = []
            levels = []
            names = []
            for i in range(self.nlevels):
                level_values = self.levels[i]
                for mi in other:
                    level_values = level_values.union(mi.levels[i])
                level_codes = [
                    recode_for_categories(
                        mi.codes[i], mi.levels[i], level_values, copy=False
                    )
                    for mi in ([self, *other])
                ]
                level_name = self.names[i]
                if any(mi.names[i] != level_name for mi in other):
                    level_name = None
                codes.append(np.concatenate(level_codes))
                levels.append(level_values)
                names.append(level_name)
            return MultiIndex(
                codes=codes, levels=levels, names=names, verify_integrity=False
            )

        to_concat = (self._values, *tuple(k._values for k in other))
        new_tuples = np.concatenate(to_concat)

        # if all(isinstance(x, MultiIndex) for x in other):
        try:
            # We only get here if other contains at least one index with tuples,
            # setting names to None automatically
            return MultiIndex.from_tuples(new_tuples)
        except (TypeError, IndexError):
            return Index(new_tuples, copy=False)

    def argsort(
        self, *args, na_position: NaPosition = "last", **kwargs
    ) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Return the positions that would sort the rows. Parameters: *args and **kwargs
        # are accepted for numpy compatibility but are not used; na_position (keyword,
        # default last; first places missing entries first). The levels are first sorted
        # monotonically (a level that cannot be compared raises a type error), codes are
        # taken from the sorting-oriented categoricals, and a lexicographic sort of the
        # codes is performed. Returns an integer position array.
        # KNOWN ISSUE: the docstring says the extra positional and keyword arguments are passed to the numpy argsort, but they are ignored.
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        *args
            Passed to `numpy.ndarray.argsort`.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        **kwargs
            Passed to `numpy.ndarray.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Integer indices that would sort the index if used as
            an indexer.

        See Also
        --------
        numpy.argsort : Similar method for NumPy arrays.
        Index.argsort : Similar method for Index.

        Examples
        --------
        >>> midx = pd.MultiIndex.from_arrays([[3, 2], ["e", "c"]])
        >>> midx
        MultiIndex([(3, 'e'),
                    (2, 'c')],
                   )

        >>> order = midx.argsort()
        >>> order
        array([1, 0])

        >>> midx[order]
        MultiIndex([(2, 'c'),
                    (3, 'e')],
                  )

        >>> midx = pd.MultiIndex.from_arrays([[2, 2], [np.nan, 0]])
        >>> midx.argsort(na_position="first")
        array([0, 1])

        >>> midx.argsort()
        array([1, 0])
        """
        target = self._sort_levels_monotonic(raise_if_incomparable=True)
        keys = [lev.codes for lev in target._get_codes_for_sorting()]
        return lexsort_indexer(keys, na_position=na_position, codes_given=True)

    def repeat(self, repeats: int, axis=None) -> MultiIndex:
        # [FUNCTION DOC]
        # Repeat each row. Parameters: repeats (an int or array of ints, converted to
        # platform integers) and axis (must be none; validated for numpy compatibility).
        # Every code array is repeated accordingly. The result shares levels, names and
        # sortorder (preserved, since repetition keeps ordering) and is not verified.
        # Returns the new MultiIndex.
        """
        Repeat elements of a MultiIndex.

        Returns a new MultiIndex where each element of the current MultiIndex
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            MultiIndex.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        MultiIndex
            Newly created MultiIndex with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays([["a", "b", "c"], [1, 2, 3]])
        >>> idx
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('c', 3)],
                   )
        >>> idx.repeat(2)
        MultiIndex([('a', 1),
                    ('a', 1),
                    ('b', 2),
                    ('b', 2),
                    ('c', 3),
                    ('c', 3)],
                   )
        >>> idx.repeat([1, 2, 3])
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('b', 2),
                    ('c', 3),
                    ('c', 3),
                    ('c', 3)],
                   )
        """
        nv.validate_repeat((), {"axis": axis})
        # error: Incompatible types in assignment (expression has type "ndarray",
        # variable has type "int")
        repeats = ensure_platform_int(repeats)  # type: ignore[assignment]
        return MultiIndex(
            levels=self.levels,
            codes=[
                level_codes.view(np.ndarray).astype(np.intp, copy=False).repeat(repeats)
                for level_codes in self.codes
            ],
            names=self.names,
            sortorder=self.sortorder,
            verify_integrity=False,
        )

    # error: Signature of "drop" incompatible with supertype "Index"
    def drop(  # type: ignore[override]
        self,
        codes,
        level: Index | np.ndarray | Iterable[Hashable] | None = None,
        errors: IgnoreRaise = "raise",
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # Return a new MultiIndex with the given labels removed. Parameters: codes
        # (array-like; tuples if level is not given, otherwise labels of that level);
        # level (default none; a level number or name); errors (default raise; with ignore
        # labels that are missing are skipped). If level is given the work is done by the
        # level-specific helper. Otherwise the labels are converted to an object array
        # where possible and each label is looked up with the location method: integers
        # give a single row, slices expand to the row range, and boolean masks expand to
        # all matching rows (a performance warning is issued if no leading level is
        # lexsorted and the performance-warnings option is on). Any other lookup result
        # type raises an assertion error. A key error from a missing label is re-raised
        # unless errors is ignore. The rows found are deleted from the index and the
        # result returned.
        """
        Make a new :class:`pandas.MultiIndex` with the passed list of codes deleted.

        This method allows for the removal of specified labels from a MultiIndex.
        The labels to be removed can be provided as a list of tuples if no level
        is specified, or as a list of labels from a specific level if the level
        parameter is provided. This can be useful for refining the structure of a
        MultiIndex to fit specific requirements.

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples when ``level`` is not specified.
        level : int or level name, default None
            Level from which the labels will be dropped.
        errors : str, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        MultiIndex
            A new MultiIndex with the specified labels removed.

        See Also
        --------
        MultiIndex.remove_unused_levels : Create new MultiIndex from current that
            removes unused levels.
        MultiIndex.reorder_levels : Rearrange levels using input order.
        MultiIndex.rename : Rename levels in a MultiIndex.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product(
        ...     [(0, 1, 2), ("green", "purple")], names=["number", "color"]
        ... )
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.drop([(1, "green"), (2, "purple")])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1, 'purple'),
                    (2,  'green')],
                   names=['number', 'color'])

        We can also drop from a specific level.

        >>> idx.drop("green", level="color")
        MultiIndex([(0, 'purple'),
                    (1, 'purple'),
                    (2, 'purple')],
                   names=['number', 'color'])

        >>> idx.drop([1, 2], level=0)
        MultiIndex([(0,  'green'),
                    (0, 'purple')],
                   names=['number', 'color'])
        """
        if level is not None:
            return self._drop_from_level(codes, level, errors)

        if not isinstance(codes, (np.ndarray, Index)):
            try:
                codes = com.index_labels_to_array(codes, dtype=np.dtype("object"))
            except ValueError:
                pass

        inds = []
        for level_codes in codes:
            try:
                loc = self.get_loc(level_codes)
                # get_loc returns either an integer, a slice, or a boolean
                # mask
                if isinstance(loc, int):
                    inds.append(loc)
                elif isinstance(loc, slice):
                    step = loc.step if loc.step is not None else 1
                    inds.extend(range(loc.start, loc.stop, step))
                elif com.is_bool_indexer(loc):
                    if get_option("performance_warnings") and self._lexsort_depth == 0:
                        warnings.warn(
                            "dropping on a non-lexsorted multi-index "
                            "without a level parameter may impact performance.",
                            PerformanceWarning,
                            stacklevel=find_stack_level(),
                        )
                    loc = loc.nonzero()[0]
                    inds.extend(loc)
                else:
                    msg = f"unsupported indexer of type {type(loc)}"
                    raise AssertionError(msg)
            except KeyError:
                if errors != "ignore":
                    raise

        return self.delete(inds)

    def _drop_from_level(
        self, codes, level, errors: IgnoreRaise = "raise"
    ) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: drops labels from one level. Parameters: codes (labels to drop), level
        # (number or name) and errors (default raise). The labels are looked up in the
        # level; labels that are absent (but not missing-value markers) are flagged, as
        # are missing markers when the level has as many entries as the index has rows.
        # Raises a key error listing the labels not found unless errors is ignore. Returns
        # the index restricted to rows whose code at this level is not among the found
        # positions.
        codes = com.index_labels_to_array(codes)
        i = self._get_level_number(level)
        index = self.levels[i]
        values = index.get_indexer(codes)
        # If nan should be dropped it will equal -1 here. We have to check which values
        # are not nan and equal -1, this means they are missing in the index
        nan_codes = isna(codes)
        values[(np.equal(nan_codes, False)) & (values == -1)] = -2
        if index.shape[0] == self.shape[0]:
            values[np.equal(nan_codes, True)] = -2

        not_found = codes[values == -2]
        if len(not_found) != 0 and errors != "ignore":
            raise KeyError(f"labels {not_found} not found in level")
        mask = ~algos.isin(self.codes[i], values)

        return self[mask]

    def swaplevel(self, i=-2, j=-1) -> MultiIndex:
        # [FUNCTION DOC]
        # Swap two levels. Parameters: i and j (level numbers or names, may be mixed;
        # defaults minus two and minus one, the last two levels). Lookup errors propagate
        # from the level-number helper. Returns a new MultiIndex with the levels, codes
        # and names of the two positions exchanged; row order is unchanged; it is
        # built without verification and without sortorder.
        """
        Swap level i with level j.

        Calling this method does not change the ordering of the values.

        Default is to swap the last two levels of the MultiIndex.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed. If i is a negative int, the first
            level is indexed relative to the end of the MultiIndex.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed. If j is a negative int, the second
            level is indexed relative to the end of the MultiIndex.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        See Also
        --------
        Series.swaplevel : Swap levels i and j in a MultiIndex.
        DataFrame.swaplevel : Swap levels i and j in a MultiIndex on a
            particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex(
        ...     levels=[["a", "b"], ["bb", "aa"], ["aaa", "bbb"]],
        ...     codes=[[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]],
        ... )
        >>> mi
        MultiIndex([('a', 'bb', 'bbb'),
                    ('a', 'aa', 'aaa'),
                    ('b', 'bb', 'bbb'),
                    ('b', 'aa', 'aaa')],
                   )
        >>> mi.swaplevel()
        MultiIndex([('a', 'bbb', 'bb'),
                    ('a', 'aaa', 'aa'),
                    ('b', 'bbb', 'bb'),
                    ('b', 'aaa', 'aa')],
                   )
        >>> mi.swaplevel(0)
        MultiIndex([('bbb', 'bb', 'a'),
                    ('aaa', 'aa', 'a'),
                    ('bbb', 'bb', 'b'),
                    ('aaa', 'aa', 'b')],
                   )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a', 'bbb'),
                    ('aa', 'a', 'aaa'),
                    ('bb', 'b', 'bbb'),
                    ('aa', 'b', 'aaa')],
                )
        """
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)

        i = self._get_level_number(i)
        j = self._get_level_number(j)

        new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
        new_codes[i], new_codes[j] = new_codes[j], new_codes[i]
        new_names[i], new_names[j] = new_names[j], new_names[i]

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    def reorder_levels(self, order) -> MultiIndex:
        # [FUNCTION DOC]
        # Reorder the levels. Parameter order is a list of level numbers or names that
        # must mention every level exactly once; each is resolved to a number and the
        # work is passed to the numeric version. Returns the new MultiIndex.
        """
        Rearrange levels using input order. May not drop or duplicate levels.

        `reorder_levels` is useful when you need to change the order of levels in
        a MultiIndex, such as when reordering levels for hierarchical indexing. It
        maintains the integrity of the MultiIndex, ensuring that all existing levels
        are present and no levels are duplicated. This method is helpful for aligning
        the index structure with other data structures or for optimizing the order
        for specific data operations.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).

        Returns
        -------
        MultiIndex
            A new MultiIndex with levels rearranged according to the specified order.

        See Also
        --------
        MultiIndex.swaplevel : Swap two levels of the MultiIndex.
        MultiIndex.set_names : Set names for the MultiIndex levels.
        DataFrame.reorder_levels : Reorder levels in a DataFrame with a MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["x", "y"])
        >>> mi
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.reorder_levels(order=[1, 0])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])

        >>> mi.reorder_levels(order=["y", "x"])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])
        """
        order = [self._get_level_number(i) for i in order]
        result = self._reorder_ilevels(order)
        return result

    def _reorder_ilevels(self, order) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: reorders levels by numeric positions. Parameter order is a list of
        # level numbers. Raises an assertion error when its length differs from the number
        # of levels (duplicates are not checked). Returns a new MultiIndex with levels,
        # codes and names rearranged, unverified and without sortorder.
        if len(order) != self.nlevels:
            raise AssertionError(
                f"Length of order must be same as number of levels ({self.nlevels}), "
                f"got {len(order)}"
            )
        new_levels = [self.levels[i] for i in order]
        new_codes = [self.codes[i] for i in order]
        new_names = [self.names[i] for i in order]

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    def _recode_for_new_levels(
        self, new_levels, *, copy: bool
    ) -> Generator[np.ndarray]:
        # [FUNCTION DOC]
        # HELPER: generator recoding codes for replacement levels. Parameters:
        # new_levels (list of levels, at most as many as this index has; an assertion
        # error otherwise) and copy (keyword-only, whether recoding must copy). Yields,
        # for each given level in order, the code array mapped from the old level
        # entries to the positions in the new level, with minus one for entries not found.
        if len(new_levels) > self.nlevels:
            raise AssertionError(
                f"Length of new_levels ({len(new_levels)}) "
                f"must be <= self.nlevels ({self.nlevels})"
            )
        for i in range(len(new_levels)):
            yield recode_for_categories(
                self.codes[i], self.levels[i], new_levels[i], copy=copy
            )

    def _get_codes_for_sorting(self) -> list[Categorical]:
        # [FUNCTION DOC]
        # HELPER: converts each code array to a categorical for sorting. The categories
        # are all integer positions up to the largest code present (not just the observed
        # ones) with ordering enabled; missing codes (minus one) stay missing so they
        # sort as missing. No validation is done. Returns a list of categoricals, one
        # per level.
        """
        we are categorizing our codes by using the
        available categories (all, not just observed)
        excluding any missing ones (-1); this is in preparation
        for sorting, where we need to disambiguate that -1 is not
        a valid valid
        """

        def cats(level_codes: np.ndarray) -> np.ndarray:
            # [FUNCTION DOC]
            # HELPER: builds the category array for one code array: the integers from zero to
            # the largest code, in the code array's dtype; empty when the codes are empty.
            return np.arange(
                level_codes.max() + 1 if len(level_codes) else 0,
                dtype=level_codes.dtype,
            )

        return [
            Categorical.from_codes(level_codes, cats(level_codes), True, validate=False)
            for level_codes in self.codes
        ]

    def sortlevel(
        self,
        level: IndexLabel = 0,
        ascending: bool | list[bool] = True,
        sort_remaining: bool = True,
        na_position: str = "first",
    ) -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # Sort by one or more levels. Parameters: level (int, name or list of them;
        # default 0); ascending (bool or a list of bools of the same length as level;
        # default true); sort_remaining (default true: after the chosen levels, sort by
        # all the other levels in order; only applies when ascending is not a list);
        # na_position (first or last; default first). Raises a value error if ascending is
        # a list whose length differs from the number of levels. Computes a lexicographic
        # sort of the code arrays and applies it to the codes; the result keeps the
        # levels and names. Its sortorder is the first sorting level when remaining
        # levels are not sorted and ascending is not a list, otherwise none. Returns a
        # pair of the sorted index and the integer position array that produced it.
        """
        Sort MultiIndex at the requested level.

        This method is useful when dealing with MultiIndex objects, allowing for
        sorting at a specific level of the index. The function preserves the
        relative ordering of data within the same level while sorting
        the overall MultiIndex. The method provides flexibility with the `ascending`
        parameter to define the sort order and with the `sort_remaining` parameter to
        control whether the remaining levels should also be sorted. Sorting a
        MultiIndex can be crucial when performing operations that require ordered
        indices, such as grouping or merging datasets. The `na_position` argument is
        important in handling missing values consistently across different levels.

        Parameters
        ----------
        level : list-like, int or str, default 0
            If a string is given, must be a name of the level.
            If list-like must be names or ints of levels.
        ascending : bool, default True
            False to sort in descending order.
            Can also be a list to specify a directed ordering.
        sort_remaining : bool, default True
            If True, sorts by the remaining levels after sorting by the specified
            `level`.
        na_position : {'first' or 'last'}, default 'first'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 2.1.0

        Returns
        -------
        sorted_index : pd.MultiIndex
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.

        See Also
        --------
        MultiIndex : A multi-level, or hierarchical, index object for pandas objects.
        Index.sort_values : Sort Index values.
        DataFrame.sort_index : Sort DataFrame by the index.
        Series.sort_index : Sort Series by the index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[0, 0], [2, 1]])
        >>> mi
        MultiIndex([(0, 2),
                    (0, 1)],
                   )

        >>> mi.sortlevel()
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(sort_remaining=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))

        >>> mi.sortlevel(1)
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(1, ascending=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))
        """
        if not is_list_like(level):
            level = [level]
        # error: Item "Hashable" of "Union[Hashable, Sequence[Hashable]]" has
        # no attribute "__iter__" (not iterable)
        level = [
            self._get_level_number(lev)
            for lev in level  # type: ignore[union-attr]
        ]
        sortorder = None

        codes = [self.codes[lev] for lev in level]
        # we have a directed ordering via ascending
        if isinstance(ascending, list):
            if not len(level) == len(ascending):
                raise ValueError("level must have same length as ascending")
        elif sort_remaining:
            codes.extend(
                [self.codes[lev] for lev in range(len(self.levels)) if lev not in level]
            )
        else:
            sortorder = level[0]

        indexer = lexsort_indexer(
            codes, orders=ascending, na_position=na_position, codes_given=True
        )

        indexer = ensure_platform_int(indexer)
        new_codes = [level_codes.take(indexer) for level_codes in self.codes]

        new_index = MultiIndex(
            codes=new_codes,
            levels=self.levels,
            names=self.names,
            sortorder=sortorder,
            verify_integrity=False,
        )

        return new_index, indexer

    def _wrap_reindex_result(self, target, indexer, preserve_names: bool):
        # [FUNCTION DOC]
        # HELPER: post-processes the result of a reindex. Parameters: target (the
        # requested new index, any kind), indexer (positions or none) and preserve_names
        # (bool). If the target is not a MultiIndex: no indexer means this index itself;
        # an indexer without missing positions means this index taken at those
        # positions; otherwise the target is converted from tuples, and if that fails
        # with a type error (not all tuples) the target is returned unchanged without
        # name handling. Then the names are possibly carried over. Returns the index.
        if not isinstance(target, MultiIndex):
            if indexer is None:
                target = self
            elif (indexer >= 0).all():
                target = self.take(indexer)
            else:
                try:
                    target = MultiIndex.from_tuples(target)
                except TypeError:
                    # not all tuples, see test_constructor_dict_multiindex_reindex_flat
                    return target

        target = self._maybe_preserve_names(target, preserve_names)
        return target

    def _maybe_preserve_names(self, target: IndexT, preserve_names: bool) -> IndexT:
        # [FUNCTION DOC]
        # HELPER: carries this index's names over to a target. Parameters: target (an
        # index) and preserve_names (bool). Only when preserve_names is set, the target has
        # the same number of levels and different names, a shallow copy of the target
        # receives this index's names. Returns the target or the renamed copy.
        if (
            preserve_names
            and target.nlevels == self.nlevels
            and target.names != self.names
        ):
            target = target.copy(deep=False)
            target.names = self.names
        return target

    # --------------------------------------------------------------------
    # Indexing Methods

    def _check_indexing_error(self, key) -> None:
        # [FUNCTION DOC]
        # HELPER: validates a key for lookups. Raises an invalid-index error when the
        # key is not hashable or is an iterator (generators are hashable and are excluded
        # explicitly). Tuples are allowed, unlike the base class. Returns nothing.
        if not is_hashable(key) or is_iterator(key):
            # We allow tuples if they are hashable, whereas other Index
            #  subclasses require scalar.
            # We have to explicitly exclude generators, as these are hashable.
            raise InvalidIndexError(key)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached property saying whether integer keys should be treated as
        # positional (GH#33355); taken from the first level.
        """
        Should integer key(s) be treated as positional?
        """
        # GH#33355
        return self.levels[0]._should_fallback_to_positional

    def _get_indexer_strict(
        self, key, axis_name: str
    ) -> tuple[Index, npt.NDArray[np.intp]]:
        # [FUNCTION DOC]
        # HELPER: strict indexer for label lists. Parameters: key (list-like of labels)
        # and axis_name (used in error messages). Non-index keys are converted to
        # object arrays preserving tuples. If the first element is not a tuple, the keys
        # are matched against level zero only; missing keys raise via the missing check,
        # and the result is this index taken at the indexer, together with the indexer.
        # Otherwise defers to the base class. Returns a pair of the target index and the
        # integer position array.
        keyarr = key
        if not isinstance(keyarr, Index):
            keyarr = com.asarray_tuplesafe(keyarr)

        if len(keyarr) and not isinstance(keyarr[0], tuple):
            indexer = self._get_indexer_level_0(keyarr)

            self._raise_if_missing(key, indexer, axis_name)
            return self[indexer], indexer

        return super()._get_indexer_strict(key, axis_name)

    def _raise_if_missing(self, key, indexer, axis_name: str) -> None:
        # [FUNCTION DOC]
        # HELPER: raises a key error for missing labels. Parameters: key, indexer (with
        # minus one for missing) and axis_name. Mirrors the first-level-only case of the
        # strict indexer: if some positions are missing, the error lists the keys that are
        # not present in level zero; if all keys are in the level but unused by the rows,
        # the error lists all the keys. For tuple keys defers to the base class.
        # Returns nothing in the non-raising path.
        keyarr = key
        if not isinstance(key, Index):
            keyarr = com.asarray_tuplesafe(key)

        if len(keyarr) and not isinstance(keyarr[0], tuple):
            # i.e. same condition for special case in MultiIndex._get_indexer_strict

            mask = indexer == -1
            if mask.any():
                check = self.levels[0].get_indexer(keyarr)
                cmask = check == -1
                if cmask.any():
                    raise KeyError(f"{keyarr[cmask]} not in index")
                # We get here when levels still contain values which are not
                # actually in Index anymore
                raise KeyError(f"{keyarr} not in index")
        else:
            return super()._raise_if_missing(key, indexer, axis_name)

    def _get_indexer_level_0(self, target) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: positions of target labels against the first level only. Parameter target
        # is list-like. Builds a categorical index from level zero and its codes, then
        # uses its lookup that allows duplicates. Returns an integer position array.
        """
        Optimized equivalent to `self.get_level_values(0).get_indexer_for(target)`.
        """
        lev = self.levels[0]
        codes = self._codes[0]
        cat = Categorical.from_codes(codes=codes, categories=lev, validate=False)
        ci = Index(cat, copy=False)
        return ci.get_indexer_for(target)

    def get_slice_bound(
        self,
        label: Hashable | Sequence[Hashable],
        side: Literal["left", "right"],
    ) -> int:
        # [FUNCTION DOC]
        # Compute the slice bound for a label in a lexsorted index. Parameters: label (a
        # label or tuple of labels for the leading levels) and side (left or right).
        # A non-tuple is treated as a one-element tuple. Delegates to the partial tuple
        # search (which may raise an unsorted-index error or a type error) and unboxes a
        # numpy scalar result. Returns the integer position: the first matching row for
        # left, one past the last for right.
        """
        For an ordered MultiIndex, compute slice bound
        that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if `side=='right') position
        of given label.

        Parameters
        ----------
        label : object or tuple of objects
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.

        Notes
        -----
        This method only works if level 0 index of the MultiIndex is lexsorted.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abbc"), list("gefd")])

        Get the locations from the leftmost 'b' in the first level
        until the end of the multiindex:

        >>> mi.get_slice_bound("b", side="left")
        1

        Like above, but if you get the locations from the rightmost
        'b' in the first level and 'f' in the second level:

        >>> mi.get_slice_bound(("b", "f"), side="right")
        3

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        if not isinstance(label, tuple):
            label = (label,)
        result = self._partial_tup_index(label, side=side)
        result = maybe_unbox_numpy_scalar(result)
        return result

    def slice_locs(self, start=None, end=None, step=None) -> tuple[int, int]:
        # [FUNCTION DOC]
        # Compute the slice positions for start and end labels (which may be partial
        # tuples). Parameters: start and end (default none: the beginning and end) and
        # step (default none). The base class implementation does all the work, using the
        # bound method of this class; this override exists to attach documentation.
        # Returns a pair of integers.
        """
        For an ordered MultiIndex, compute the slice locations for input
        labels.

        The input labels can be tuples representing partial levels, e.g. for a
        MultiIndex with 3 levels, you can pass a single value (corresponding to
        the first level), or a 1-, 2-, or 3-tuple.

        Parameters
        ----------
        start : label or tuple, default None
            If None, defaults to the beginning
        end : label or tuple
            If None, defaults to the end
        step : int or None
            Slice step

        Returns
        -------
        (start, end) : (int, int)

        Notes
        -----
        This method only works if the MultiIndex is properly lexsorted. So,
        if only the first 2 levels of a 3-level MultiIndex are lexsorted,
        you can only pass two levels to ``.slice_locs``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ...     [list("abbd"), list("deff")], names=["A", "B"]
        ... )

        Get the slice locations from the beginning of 'b' in the first level
        until the end of the multiindex:

        >>> mi.slice_locs(start="b")
        (1, 4)

        Like above, but stop at the end of 'b' in the first level and 'f' in
        the second level:

        >>> mi.slice_locs(start="b", end=("b", "f"))
        (1, 3)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        # This function adds nothing to its parent implementation (the magic
        # happens in get_slice_bound method), but it adds meaningful doc.
        return super().slice_locs(start, end, step)

    def _partial_tup_index(self, tup: tuple, side: Literal["left", "right"] = "left"):
        # [FUNCTION DOC]
        # HELPER: finds a bound for a partial key. Parameters: tup (tuple of labels for
        # the leading levels) and side (left or right; default left). Raises an
        # unsorted-index error if the tuple is longer than the lexsort depth, and a
        # type error for labels that cannot be compared to a level. Narrows a window of
        # rows level by level using binary search on the codes: labels missing from the
        # level are located by searching the level and returning immediately; present
        # labels narrow the window (or return the bound at the last level). Returns the
        # integer bound.
        if len(tup) > self._lexsort_depth:
            raise UnsortedIndexError(
                f"Key length ({len(tup)}) was greater than MultiIndex lexsort depth "
                f"({self._lexsort_depth})"
            )

        n = len(tup)
        start, end = 0, len(self)
        zipped = zip(tup, self.levels, self.codes, strict=True)
        for k, (lab, lev, level_codes) in enumerate(zipped):
            section = level_codes[start:end]

            loc: npt.NDArray[np.intp] | np.intp | int
            if lab not in lev and not isna(lab):
                # short circuit
                try:
                    loc = algos.searchsorted(lev, lab, side=side)
                except TypeError as err:
                    # non-comparable e.g. test_slice_locs_with_type_mismatch
                    raise TypeError(f"Level type mismatch: {lab}") from err
                if not is_integer(loc):
                    # non-comparable level, e.g. test_groupby_example
                    raise TypeError(f"Level type mismatch: {lab}")
                if side == "right" and loc >= 0:
                    loc -= 1
                return start + algos.searchsorted(section, loc, side=side)

            idx = self._get_loc_single_level_index(lev, lab)
            if isinstance(idx, slice) and k < n - 1:
                # Get start and end value from slice, necessary when a non-integer
                # interval is given as input GH#37707
                start = idx.start
                end = idx.stop
            elif k < n - 1:
                # error: Incompatible types in assignment (expression has type
                # "Union[ndarray[Any, dtype[signedinteger[Any]]]
                end = start + algos.searchsorted(  # type: ignore[assignment]
                    section, idx, side="right"
                )
                # error: Incompatible types in assignment (expression has type
                # "Union[ndarray[Any, dtype[signedinteger[Any]]]
                start = start + algos.searchsorted(  # type: ignore[assignment]
                    section, idx, side="left"
                )
            elif isinstance(idx, slice):
                idx = idx.start
                return start + algos.searchsorted(section, idx, side=side)
            else:
                return start + algos.searchsorted(section, idx, side=side)

    def _get_loc_single_level_index(self, level_index: Index, key: Hashable) -> int:
        # [FUNCTION DOC]
        # HELPER: location of a key in a single level's index. Parameters: level_index and
        # key. If the key is a scalar missing value the location is minus one (the code
        # for missing); otherwise the level's own location lookup is used, which may
        # return an integer or a slice, or raise a key error.
        # KNOWN ISSUE: the return annotation and docstring say the result is an integer, but the level lookup may return a slice, and callers handle slices.
        """
        If key is NA value, location of index unify as -1.

        Parameters
        ----------
        level_index: Index
        key : label

        Returns
        -------
        loc : int
            If key is NA value, loc is -1
            Else, location of key in index.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        """
        if is_scalar(key) and isna(key):
            # TODO: need is_valid_na_for_dtype(key, level_index.dtype)
            return -1
        else:
            return level_index.get_loc(key)

    def get_loc(self, key):
        # [FUNCTION DOC]
        # Return the location of a label or a tuple of labels. Parameter key: a single
        # label (matched against level zero) or a tuple with at most one label per level.
        # Returns an integer, a slice, or a boolean mask. Raises a key error when the
        # tuple is longer than the number of levels or nothing matches; raises an
        # invalid-index error from the indexing check for unhashable or iterator keys.
        # Behavior: a non-tuple uses the level-zero indexer and converts an index array to a
        # slice or mask. A full-length tuple on a unique index goes to the engine (on a type
        # error, such as partial string slicing, it falls back to the level-wise lookup).
        # Otherwise the key is split into a part within the lexsort depth, resolved with a
        # slice, and a remainder that is filtered level by level over that slice (a
        # performance warning is emitted when the remainder is non-empty and the option is
        # on). The result is a slice when the filter keeps everything.
        """
        Get location for a label or a tuple of labels. The location is returned \
        as an integer/slice or boolean mask.

        This method returns the integer location, slice object, or boolean mask
        corresponding to the specified key, which can be a single label or a tuple
        of labels. The key represents a position in the MultiIndex, and the location
        indicates where the key is found within the index.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)
            A label or tuple of labels that correspond to the levels of the MultiIndex.
            The key must match the structure of the MultiIndex.

        Returns
        -------
        int, slice object or boolean mask
            If the key is past the lexsort depth, the return may be a
            boolean mask array, otherwise it is always a slice or int.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Notes
        -----
        The key cannot be a slice, list of same-level labels, a boolean mask,
        or a sequence of such. If you want to use those, use
        :meth:`MultiIndex.get_locs` instead.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abb"), list("def")])

        >>> mi.get_loc("b")
        slice(1, 3, None)

        >>> mi.get_loc(("b", "e"))
        1
        """
        self._check_indexing_error(key)

        def _maybe_to_slice(loc):
            # [FUNCTION DOC]
            # HELPER: converts an array of integer positions into a slice if the positions are
            # contiguous, else into a boolean mask over the whole index. Non-array input or
            # arrays of other dtypes are returned unchanged.
            """convert integer indexer to boolean mask or slice if possible"""
            if not isinstance(loc, np.ndarray) or loc.dtype != np.intp:
                return loc

            loc = lib.maybe_indices_to_slice(loc, len(self))
            if isinstance(loc, slice):
                return loc

            mask = np.empty(len(self), dtype="bool")
            mask.fill(False)
            mask[loc] = True
            return mask

        if not isinstance(key, tuple):
            loc = self._get_level_indexer(key, level=0)
            return _maybe_to_slice(loc)

        keylen = len(key)
        if self.nlevels < keylen:
            raise KeyError(
                f"Key length ({keylen}) exceeds index depth ({self.nlevels})"
            )

        if keylen == self.nlevels and self.is_unique:
            # TODO: what if we have an IntervalIndex level?
            #  i.e. do we need _index_as_unique on that level?
            try:
                return self._engine.get_loc(key)
            except KeyError as err:
                raise KeyError(key) from err
            except TypeError:
                # e.g. test_partial_slicing_with_multiindex partial string slicing
                loc, _ = self.get_loc_level(key, range(self.nlevels))
                return loc

        # -- partial selection or non-unique index
        # break the key into 2 parts based on the lexsort_depth of the index;
        # the first part returns a continuous slice of the index; the 2nd part
        # needs linear search within the slice
        i = self._lexsort_depth
        lead_key, follow_key = key[:i], key[i:]

        if not lead_key:
            start = 0
            stop = len(self)
        else:
            try:
                start, stop = self.slice_locs(lead_key, lead_key)
            except TypeError as err:
                # e.g. test_groupby_example key = ((0, 0, 1, 2), "new_col")
                #  when self has 5 integer levels
                raise KeyError(key) from err

        if start == stop:
            raise KeyError(key)

        if not follow_key:
            return slice(start, stop)

        if get_option("performance_warnings"):
            warnings.warn(
                "indexing past lexsort depth may impact performance.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        loc = np.arange(start, stop, dtype=np.intp)

        for i, k in enumerate(follow_key, len(lead_key)):
            mask = self.codes[i][loc] == self._get_loc_single_level_index(
                self.levels[i], k
            )
            if not mask.all():
                loc = loc[mask]
            if not len(loc):
                raise KeyError(key)

        return _maybe_to_slice(loc) if len(loc) != stop - start else slice(start, stop)

    def get_loc_level(self, key, level: IndexLabel = 0, drop_level: bool = True):
        # [FUNCTION DOC]
        # Get the locations and the reduced index for a key. Parameters: key (a label,
        # a tuple, or a list of labels); level (a level number or name, or a
        # range/list/tuple of them; default 0; names are resolved to numbers); drop_level
        # (default true: the selected levels are removed from the returned index; when
        # false the returned index keeps all levels and is the index sliced at the
        # location). Returns a pair: the location (integer, slice or boolean mask) and the
        # reduced index (an Index or MultiIndex, or none when the key covers every
        # level). Errors from the helper propagate (key errors, assertion errors for
        # mismatched key and level lengths, type errors for unsupported keys).
        """
        Get location and sliced index for requested label(s)/level(s).

        The `get_loc_level` method is a more advanced form of `get_loc`, allowing
        users to specify not just a label or sequence of labels, but also the level(s)
        in which to search. This method is useful when you need to isolate particular
        sections of a MultiIndex, either for further analysis or for slicing and
        dicing the data. The method provides flexibility in terms of maintaining
        or dropping levels from the resulting index based on the `drop_level`
        parameter.

        Parameters
        ----------
        key : label or sequence of labels
            The label(s) for which to get the location.
        level : int/level name or list thereof, optional
            The level(s) in the MultiIndex to consider. If not provided, defaults
            to the first level.
        drop_level : bool, default True
            If ``False``, the resulting index will not drop any level.

        Returns
        -------
        tuple
            A 2-tuple where the elements :

            Element 0: int, slice object or boolean array.

            Element 1: The resulting sliced multiindex/index. If the key
            contains all levels, this will be ``None``.

        See Also
        --------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abb"), list("def")], names=["A", "B"])

        >>> mi.get_loc_level("b")
        (slice(1, 3, None), Index(['e', 'f'], dtype='str', name='B'))

        >>> mi.get_loc_level("e", level="B")
        (array([False,  True, False]), Index(['b'], dtype='str', name='A'))

        >>> mi.get_loc_level(["b", "e"])
        (1, None)
        """
        if not isinstance(level, (range, list, tuple)):
            level = self._get_level_number(level)
        else:
            level = [self._get_level_number(lev) for lev in level]

        loc, mi = self._get_loc_level(key, level=level)
        if not drop_level:
            if lib.is_integer(loc):
                # Slice index must be an integer or None
                mi = self[loc : loc + 1]
            else:
                mi = self[loc]
        return loc, mi

    def _get_loc_level(self, key, level: int | list[int] = 0):
        # [FUNCTION DOC]
        # HELPER: core of the location-and-reduced-index lookup with level numbers
        # already resolved. Parameters: key (label, tuple, or list) and level (an integer
        # or a list/tuple of integers; default 0). Returns the pair (location, reduced
        # index). Cases: for several levels, the key must have the same count (assertion
        # error otherwise), each is resolved recursively and the boolean masks are
        # combined with logical and, then the selected levels are dropped (if that would
        # drop all levels the unreduced selection is returned instead). A list key is
        # treated as a tuple (kludge for #1796). A tuple key at level zero is first tried
        # as a single label of level zero; otherwise, if it has no slices, a full-length
        # key on a unique index goes straight to the engine, else partial selection uses
        # the location method and drops the levels given by the key (with special
        # handling when every level is given, and for partial string indexing); a tuple
        # containing slices intersects per-level indexers. A non-tuple key is resolved
        # on the single level, and for partial string keys on datetime-like levels the
        # index is returned without dropping. A key error comes out of lookups; a type
        # error is raised when a tuple element is neither a label nor a slice.
        """
        get_loc_level but with `level` known to be positional, not name-based.
        """

        # different name to distinguish from maybe_droplevels
        def maybe_mi_droplevels(indexer, levels):
            # [FUNCTION DOC]
            # HELPER: slices the index by an indexer and drops the given level numbers (from
            # the highest to the lowest). Parameters: indexer (integer, slice or mask) and
            # levels (iterable of level numbers). Errors that occur when a level does not
            # exist or all levels are dropped (value errors) are left to the caller. Returns
            # the reduced index.
            """
            If level does not exist or all levels were dropped, the exception
            has to be handled outside.
            """
            new_index = self[indexer]

            for i in sorted(levels, reverse=True):
                new_index = new_index._drop_level_numbers([i])

            return new_index

        if isinstance(level, (tuple, list)):
            if len(key) != len(level):
                raise AssertionError(
                    "Key for location must have same length as number of levels"
                )
            result = None
            for lev, k in zip(level, key, strict=True):
                loc, new_index = self._get_loc_level(k, level=lev)
                if isinstance(loc, slice):
                    mask = np.zeros(len(self), dtype=bool)
                    mask[loc] = True
                    loc = mask
                result = loc if result is None else result & loc

            try:
                # FIXME: we should be only dropping levels on which we are
                #  scalar-indexing
                mi = maybe_mi_droplevels(result, level)
            except ValueError:
                # droplevel failed because we tried to drop all levels,
                #  i.e. len(level) == self.nlevels
                mi = self[result]

            return result, mi

        # kludge for #1796
        if isinstance(key, list):
            key = tuple(key)

        if isinstance(key, tuple) and level == 0:
            try:
                # Check if this tuple is a single key in our first level
                if key in self.levels[0]:
                    indexer = self._get_level_indexer(key, level=level)
                    new_index = maybe_mi_droplevels(indexer, [0])
                    return indexer, new_index
            except (TypeError, InvalidIndexError):
                pass

            if not any(isinstance(k, slice) for k in key):
                if len(key) == self.nlevels and self.is_unique:
                    # Complete key in unique index -> standard get_loc
                    try:
                        return (self._engine.get_loc(key), None)
                    except KeyError as err:
                        raise KeyError(key) from err
                    except TypeError:
                        # e.g. partial string indexing
                        #  test_partial_string_timestamp_multiindex
                        pass

                # partial selection
                indexer = self.get_loc(key)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                if len(ilevels) == self.nlevels:
                    if is_integer(indexer):
                        # we are dropping all levels
                        return indexer, None

                    # TODO: in some cases we still need to drop some levels,
                    #  e.g. test_multiindex_perf_warn
                    # test_partial_string_timestamp_multiindex
                    ilevels = [
                        i
                        for i in range(len(key))
                        if (
                            not isinstance(key[i], str)
                            or not self.levels[i]._supports_partial_string_indexing
                        )
                        and key[i] != slice(None, None)
                    ]
                    if len(ilevels) == self.nlevels:
                        # TODO: why?
                        ilevels = []
                return indexer, maybe_mi_droplevels(indexer, ilevels)

            else:
                indexer = None
                for i, k in enumerate(key):
                    if not isinstance(k, slice):
                        loc_level = self._get_level_indexer(k, level=i)
                        if isinstance(loc_level, slice):
                            if com.is_null_slice(loc_level) or com.is_full_slice(
                                loc_level, len(self)
                            ):
                                # everything
                                continue

                            # e.g. test_xs_IndexSlice_argument_not_implemented
                            k_index = np.zeros(len(self), dtype=bool)
                            k_index[loc_level] = True

                        else:
                            k_index = loc_level

                    elif com.is_null_slice(k):
                        # taking everything, does not affect `indexer` below
                        continue

                    else:
                        # FIXME: this message can be inaccurate, e.g.
                        #  test_series_varied_multiindex_alignment
                        raise TypeError(f"Expected label or tuple of labels, got {key}")

                    if indexer is None:
                        indexer = k_index
                    else:
                        indexer &= k_index
                if indexer is None:
                    indexer = slice(None, None)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                return indexer, maybe_mi_droplevels(indexer, ilevels)
        else:
            indexer = self._get_level_indexer(key, level=level)
            if (
                isinstance(key, str)
                and self.levels[level]._supports_partial_string_indexing
            ):
                # check to see if we did an exact lookup vs sliced
                check = self.levels[level].get_loc(key)
                if not is_integer(check):
                    # e.g. test_partial_string_timestamp_multiindex
                    return indexer, self[indexer]

            try:
                result_index = maybe_mi_droplevels(indexer, [level])
            except ValueError:
                result_index = self[indexer]

            return indexer, result_index

    def _get_level_indexer(
        self, key, level: int = 0, indexer: npt.NDArray[np.bool_] | None = None
    ):
        # [FUNCTION DOC]
        # HELPER: locations of a key within one level of the whole index. Parameters:
        # key (a label, a slice, or a list-like label handled by the level index); level
        # (positional number only, default 0); indexer (optional boolean array restricting
        # the positions examined, default none). Returns a slice when the level is sorted
        # and the match is contiguous, otherwise a boolean array. Slices: start and stop
        # are found in the level (negative steps handled); partial string slices on
        # datetime-like levels produce slices from the level; when the level is not the
        # first or the index is not lexsorted or a step is given, a boolean mask is
        # built from the code ranges (stop made inclusive); otherwise the bounds are found
        # by binary search on the codes. Scalar keys: a missing key gives the missing
        # code; level-wise match produces a mask (error if unused) or, for a sorted first
        # level, a slice by binary search. Raises a key error if the label is in the
        # level but used by no row, or is absent.
        # `level` kwarg is _always_ positional, never name
        # return a boolean array or slice showing where the key is
        # in the totality of values
        # if the indexer is provided, then use this

        level_index = self.levels[level]
        level_codes = self.codes[level]

        def convert_indexer(start, stop, step, indexer=indexer, codes=level_codes):
            # [FUNCTION DOC]
            # HELPER: builds a boolean mask of codes within a range. Parameters: start, stop,
            # step (the code range; step none or one gives a simple range test, otherwise
            # membership in an arithmetic sequence of codes), indexer (default the enclosing
            # indexer) and codes (default the level's codes). If an existing indexer is given,
            # only its true positions are examined and the result is a copy of that indexer
            # with those positions updated. Returns the boolean mask.
            # Compute a bool indexer to identify the positions to take.
            # If we have an existing indexer, we only need to examine the
            # subset of positions where the existing indexer is True.
            if indexer is not None:
                # we only need to look at the subset of codes where the
                # existing indexer equals True
                codes = codes[indexer]

            if step is None or step == 1:
                new_indexer = (codes >= start) & (codes < stop)
            else:
                r = np.arange(start, stop, step, dtype=codes.dtype)
                new_indexer = algos.isin(codes, r)

            if indexer is None:
                return new_indexer

            indexer = indexer.copy()
            indexer[indexer] = new_indexer
            return indexer

        if isinstance(key, slice):
            # handle a slice, returning a slice if we can
            # otherwise a boolean indexer
            step = key.step
            is_negative_step = step is not None and step < 0

            try:
                if key.start is not None:
                    start = level_index.get_loc(key.start)
                elif is_negative_step:
                    start = len(level_index) - 1
                else:
                    start = 0

                if key.stop is not None:
                    stop = level_index.get_loc(key.stop)
                elif is_negative_step:
                    stop = 0
                elif isinstance(start, slice):
                    stop = len(level_index)
                else:
                    stop = len(level_index) - 1
            except KeyError:
                # we have a partial slice (like looking up a partial date
                # string)
                start = stop = level_index.slice_indexer(key.start, key.stop, key.step)
                step = start.step

            if isinstance(start, slice) or isinstance(stop, slice):
                # we have a slice for start and/or stop
                # a partial date slicer on a DatetimeIndex generates a slice
                # note that the stop ALREADY includes the stopped point (if
                # it was a string sliced)
                start = getattr(start, "start", start)
                stop = getattr(stop, "stop", stop)
                return convert_indexer(start, stop, step)

            elif level > 0 or self._lexsort_depth == 0 or step is not None:
                # need to have like semantics here to right
                # searching as when we are using a slice
                # so adjust the stop by 1 (so we include stop)
                stop = (stop - 1) if is_negative_step else (stop + 1)
                return convert_indexer(start, stop, step)
            else:
                # sorted, so can return slice object -> view
                i = algos.searchsorted(level_codes, start, side="left")
                j = algos.searchsorted(level_codes, stop, side="right")
                return slice(i, j, step)

        else:
            idx = self._get_loc_single_level_index(level_index, key)

            if level > 0 or self._lexsort_depth == 0:
                # Desired level is not sorted
                if isinstance(idx, slice):
                    # test_get_loc_partial_timestamp_multiindex
                    locs = (level_codes >= idx.start) & (level_codes < idx.stop)
                    return locs

                locs = np.asarray(level_codes == idx, dtype=bool)

                if not locs.any():
                    # The label is present in self.levels[level] but unused:
                    raise KeyError(key)
                return locs

            if isinstance(idx, slice):
                # e.g. test_partial_string_timestamp_multiindex
                start = algos.searchsorted(level_codes, idx.start, side="left")
                # NB: "left" here bc of slice semantics
                end = algos.searchsorted(level_codes, idx.stop, side="left")
            else:
                start = algos.searchsorted(level_codes, idx, side="left")
                end = algos.searchsorted(level_codes, idx, side="right")

            if start == end:
                # The label is present in self.levels[level] but unused:
                raise KeyError(key)
            return slice(maybe_unbox_numpy_scalar(start), maybe_unbox_numpy_scalar(end))

    def get_locs(self, seq) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # Get integer locations for a selection spanning levels. Parameter seq: one
        # selection per level, each being a label, slice, list of labels, boolean mask
        # or the null slice (which skips the level); a missing trailing level means
        # no restriction. Raises an unsorted-index error if a real slice is used on a
        # level beyond the lexsort depth, a not-implemented error if an ellipsis appears,
        # a value error for a boolean mask of the wrong length, and a key error if a
        # selection combined with earlier ones leaves no rows although the selection
        # alone matched something. Per-level results are combined with logical and;
        # lists are or-ed label by label (falling back to per-label lookups when the
        # list itself is not a single key). If nothing is selected an empty integer
        # array is returned; a trailing null slice on the first selection returns all
        # positions. The final positions are ordered like the selection. Returns an
        # integer position array.
        """
        Get location for a sequence of labels.

        Parameters
        ----------
        seq : label, slice, list, mask or a sequence of such
           You should use one of the above for each level.
           If a level should not be used, set it to ``slice(None)``.

        Returns
        -------
        numpy.ndarray
            NumPy array of integers suitable for passing to iloc.

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list("abb"), list("def")])

        >>> mi.get_locs("b")  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([slice(None), ["e", "f"]])  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([[True, False, True], slice("e", "f")])  # doctest: +SKIP
        array([2], dtype=int64)
        """

        # must be lexsorted to at least as many levels
        true_slices = [i for (i, s) in enumerate(com.is_true_slices(seq)) if s]
        if true_slices and true_slices[-1] >= self._lexsort_depth:
            raise UnsortedIndexError(
                "MultiIndex slicing requires the index to be lexsorted: slicing "
                f"on levels {true_slices}, lexsort depth {self._lexsort_depth}"
            )

        if any(x is Ellipsis for x in seq):
            raise NotImplementedError(
                "MultiIndex does not support indexing with Ellipsis"
            )

        n = len(self)

        def _to_bool_indexer(indexer) -> npt.NDArray[np.bool_]:
            # [FUNCTION DOC]
            # HELPER: converts a slice into a full-length boolean mask (positions in the
            # slice true) and returns other indexers unchanged.
            if isinstance(indexer, slice):
                new_indexer = np.zeros(n, dtype=np.bool_)
                new_indexer[indexer] = True
                return new_indexer
            return indexer

        # a bool indexer for the positions we want to take
        indexer: npt.NDArray[np.bool_] | None = None

        for i, k in enumerate(seq):
            lvl_indexer: npt.NDArray[np.bool_] | slice | None = None

            if com.is_bool_indexer(k):
                if len(k) != n:
                    raise ValueError(
                        "cannot index with a boolean indexer that "
                        "is not the same length as the index"
                    )
                if isinstance(k, (ABCSeries, Index)):
                    k = k._values
                lvl_indexer = np.asarray(k)
                if indexer is None:
                    lvl_indexer = lvl_indexer.copy()

            elif is_list_like(k):
                # a collection of labels to include from this level (these are or'd)

                # GH#27591 check if this is a single tuple key in the level
                try:
                    lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
                except (InvalidIndexError, TypeError, KeyError) as err:
                    # InvalidIndexError e.g. non-hashable, fall back to treating
                    #  this as a sequence of labels
                    # KeyError it can be ambiguous if this is a label or sequence
                    #  of labels
                    #  github.com/pandas-dev/pandas/issues/39424#issuecomment-871626708
                    for x in k:
                        if not is_hashable(x):
                            # e.g. slice
                            raise err
                        # GH 39424: Ignore not founds
                        # GH 42351: No longer ignore not founds & enforced in 2.0
                        # TODO: how to handle IntervalIndex level? (no test cases)
                        item_indexer = self._get_level_indexer(
                            x, level=i, indexer=indexer
                        )
                        if lvl_indexer is None:
                            lvl_indexer = _to_bool_indexer(item_indexer)
                        elif isinstance(item_indexer, slice):
                            lvl_indexer[item_indexer] = True  # type: ignore[index]
                        else:
                            lvl_indexer |= item_indexer

                if lvl_indexer is None:
                    # no matches we are done
                    # test_loc_getitem_duplicates_multiindex_empty_indexer
                    return np.array([], dtype=np.intp)

            elif com.is_null_slice(k):
                # empty slice
                if indexer is None and i == len(seq) - 1:
                    return np.arange(n, dtype=np.intp)
                continue

            else:
                # a slice or a single label
                lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)

            # update indexer
            lvl_indexer = _to_bool_indexer(lvl_indexer)
            if indexer is None:
                indexer = lvl_indexer
            else:
                indexer &= lvl_indexer
                if not np.any(indexer) and np.any(lvl_indexer):
                    raise KeyError(seq)

        # empty indexer
        if indexer is None:
            return np.array([], dtype=np.intp)

        pos_indexer = indexer.nonzero()[0]
        return self._reorder_indexer(seq, pos_indexer)

    # --------------------------------------------------------------------

    def _reorder_indexer(
        self,
        seq: tuple[Scalar | Iterable | AnyArrayLike, ...],
        indexer: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.intp]:
        # [FUNCTION DOC]
        # HELPER: reorders a position indexer so rows follow the order of the labels in
        # the selection. Parameters: seq (the per-level selections: label, slice, list,
        # mask, or a sequence of them) and indexer (integer positions). First checks
        # whether reordering is required (scalars, null slices and masks need none;
        # list-likes of more than one element need it unless the index is lexsorted and
        # the codes of the list are increasing; slices need it unless lexsorted and not
        # reversed; anything else needs it). If not needed the indexer is returned as is.
        # Otherwise a key array per level is built: mappings from level codes to the
        # order in the list (absent labels ignored, unlisted codes last), reversed
        # order for negative-step slices, constant for the null slice, and natural order
        # for others, and the indexer is sorted lexicographically by those keys (most
        # significant level first). Returns the reordered integer positions.
        """
        Reorder an indexer of a MultiIndex (self) so that the labels are in the
        same order as given in seq

        Parameters
        ----------
        seq : label/slice/list/mask or a sequence of such
        indexer: a position indexer of self

        Returns
        -------
        indexer : a sorted position indexer of self ordered as seq
        """

        # check if sorting is necessary
        need_sort = False
        for i, k in enumerate(seq):
            if com.is_null_slice(k) or com.is_bool_indexer(k) or is_scalar(k):
                pass
            elif is_list_like(k):
                if len(k) <= 1:  # type: ignore[arg-type]
                    pass
                elif self._is_lexsorted():
                    # If the index is lexsorted and the list_like label
                    # in seq are sorted then we do not need to sort
                    k_codes = self.levels[i].get_indexer(k)
                    k_codes = k_codes[k_codes >= 0]  # Filter absent keys
                    # True if the given codes are not ordered
                    need_sort = (k_codes[:-1] > k_codes[1:]).any()
                else:
                    need_sort = True
            elif isinstance(k, slice):
                if self._is_lexsorted():
                    need_sort = k.step is not None and k.step < 0
                else:
                    need_sort = True
            else:
                need_sort = True
            if need_sort:
                break
        if not need_sort:
            return indexer

        n = len(self)
        keys: tuple[np.ndarray, ...] = ()
        # For each level of the sequence in seq, map the level codes with the
        # order they appears in a list-like sequence
        # This mapping is then use to reorder the indexer
        for i, k in enumerate(seq):
            if is_scalar(k):
                # GH#34603 we want to treat a scalar the same as an all equal list
                k = [k]
            if com.is_bool_indexer(k):
                new_order = np.arange(n)[indexer]
            elif is_list_like(k):
                # Generate a map with all level codes as sorted initially
                if not isinstance(k, (np.ndarray, ExtensionArray, Index, ABCSeries)):
                    k = sanitize_array(k, None)
                k = algos.unique(k)
                key_order_map = np.ones(len(self.levels[i]), dtype=np.uint64) * len(
                    self.levels[i]
                )
                # Set order as given in the indexer list
                level_indexer = self.levels[i].get_indexer(k)
                level_indexer = level_indexer[level_indexer >= 0]  # Filter absent keys
                key_order_map[level_indexer] = np.arange(len(level_indexer))

                new_order = key_order_map[self.codes[i][indexer]]
            elif isinstance(k, slice) and k.step is not None and k.step < 0:
                # flip order for negative step
                new_order = np.arange(n - 1, -1, -1)[indexer]
            elif isinstance(k, slice) and k.start is None and k.stop is None:
                # slice(None) should not determine order GH#31330
                new_order = np.ones((n,), dtype=np.intp)[indexer]
            else:
                # For all other case, use the same order as the level
                new_order = np.arange(n)[indexer]
            keys = (new_order, *keys)

        # Find the reordering using lexsort on the keys mapping
        ind = np.lexsort(keys)
        return indexer[ind]

    def truncate(self, before=None, after=None) -> MultiIndex:
        # [FUNCTION DOC]
        # Cut the index between two labels. Parameters: before and after (labels or
        # partial tuples, default none: the start and the end). Raises a value error when
        # both are given and after sorts before before (the check is skipped when either
        # is falsy). The first level is cut with its slice locations and codes are
        # sliced by the full slice locations, with the first level codes shifted. Returns
        # a new MultiIndex that keeps the names and has no sortorder or verification.
        """
        Slice index between two labels / tuples, return new MultiIndex.

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start.
        after : label or tuple, can be partial. Default None
            None defaults to end.

        Returns
        -------
        MultiIndex
            The truncated MultiIndex.

        See Also
        --------
        DataFrame.truncate : Truncate a DataFrame before and after some index values.
        Series.truncate : Truncate a Series before and after some index values.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([["a", "b", "c"], ["x", "y", "z"]])
        >>> mi
        MultiIndex([('a', 'x'), ('b', 'y'), ('c', 'z')],
                   )
        >>> mi.truncate(before="a", after="b")
        MultiIndex([('a', 'x'), ('b', 'y')],
                   )
        """
        if after and before and after < before:
            raise ValueError("after < before")

        i, j = self.levels[0].slice_locs(before, after)
        left, right = self.slice_locs(before, after)

        new_levels = list(self.levels)
        new_levels[0] = new_levels[0][i:j]

        new_codes = [level_codes[left:right] for level_codes in self.codes]
        new_codes[0] = new_codes[0] - i

        return MultiIndex(
            levels=new_levels,
            codes=new_codes,
            names=self._names,
            verify_integrity=False,
        )

    def equals(self, other: object) -> bool:
        # [FUNCTION DOC]
        # Equality of labeling. Parameter other: any object. True if identical objects;
        # false if not an index, lengths differ, or (for non-MultiIndex input) the
        # dtype cannot be compared; for a flat index of tuples the tuple values are
        # compared. For two MultiIndexes: the number of levels must match, and for each
        # level the missing positions must coincide, the codes recoded onto this
        # level's labels must be equal, and the empty slices of the levels must be equal
        # (to distinguish dtypes). The levels themselves need not be equal. Returns a
        # boolean.
        """
        Determines if two MultiIndex objects have the same labeling information
        (the levels themselves do not necessarily have to be the same)

        See Also
        --------
        equal_levels
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        if len(self) != len(other):
            return False

        if not isinstance(other, MultiIndex):
            # d-level MultiIndex can equal d-tuple Index
            if not self._should_compare(other):
                # object Index or Categorical[object] may contain tuples
                return False
            return array_equivalent(self._values, other._values)

        if self.nlevels != other.nlevels:
            return False

        for i in range(self.nlevels):
            self_codes = self.codes[i]
            other_codes = other.codes[i]
            self_mask = self_codes == -1
            other_mask = other_codes == -1
            if not np.array_equal(self_mask, other_mask):
                return False
            self_level = self.levels[i]
            other_level = other.levels[i]
            new_codes = recode_for_categories(
                other_codes, other_level, self_level, copy=False
            )
            if not np.array_equal(self_codes, new_codes):
                return False
            if not self_level[:0].equals(other_level[:0]):
                # e.g. Int64 != int64
                return False
        return True

    def equal_levels(self, other: MultiIndex) -> bool:
        # [FUNCTION DOC]
        # Return true if every level of this index equals the corresponding level of the
        # other and the number of levels agrees.
        """
        Return True if the levels of both MultiIndex objects are the same

        """
        if self.nlevels != other.nlevels:
            return False

        for i in range(self.nlevels):
            if not self.levels[i].equals(other.levels[i]):
                return False
        return True

    # --------------------------------------------------------------------
    # Set Methods

    def _union(self, other, sort) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: union with another index. Parameters: other and sort (true, false or
        # none). The other is converted for set operations, which also gives the common
        # names. If other has duplicates the generic base implementation is used on
        # renamed versions and the result is converted back to a MultiIndex if
        # needed. Otherwise the rows of other missing here are found and appended (or
        # this index returns with reconciled names), and unless sort is false the
        # result is sorted; unorderable values give a runtime warning (or raise a type
        # error when sort is true). Returns a MultiIndex.
        other, result_names = self._convert_can_do_setop(other)
        if other.has_duplicates:
            # This is only necessary if other has dupes,
            # otherwise difference is faster
            result = super(MultiIndex, self.rename(result_names))._union(
                other.rename(result_names), sort
            )

            if isinstance(result, MultiIndex):
                return result
            return MultiIndex.from_arrays(
                zip(*result, strict=True), sortorder=None, names=result_names
            )

        else:
            right_missing = other.difference(self, sort=False)
            if len(right_missing):
                result = self.append(right_missing)
            else:
                result = self._get_reconciled_name_object(other)

            if sort is not False:
                try:
                    result = result.sort_values()
                except TypeError:
                    if sort is True:
                        raise
                    warnings.warn(
                        "The values in the array are unorderable. "
                        "Pass `sort=False` to suppress this warning.",
                        RuntimeWarning,
                        stacklevel=find_stack_level(),
                    )
            return result

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        # [FUNCTION DOC]
        # HELPER: comparability check; true only for the object dtype.
        return is_object_dtype(dtype)

    def _get_reconciled_name_object(self, other) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: returns a result of a set operation that equals this index. Gets the
        # consensus names with the other index; if they differ from this index's names a
        # renamed index is returned, otherwise a shallow copy.
        """
        If the result of a set operation will be self,
        return a shallow copy of self.
        """
        names = self._maybe_match_names(other)
        if self.names != names:
            return self.rename(names)
        return self.copy(deep=False)

    def _maybe_match_names(self, other):
        # [FUNCTION DOC]
        # HELPER: finds names common to this and another index. Parameter other is
        # another index. If the numbers of levels differ, a list of none for each level is
        # returned. Otherwise each level keeps its name only when both are equal, else
        # none (a pair of not-a-number names is not treated as equal). Returns a list.
        """
        Try to find common names to attach to the result of an operation between
        a and b. Return a consensus list of names if they match at least partly
        or list of None if they have completely different names.
        """
        if len(self.names) != len(other.names):
            return [None] * len(self.names)
        names = []
        for a_name, b_name in zip(self.names, other.names, strict=True):
            if a_name == b_name:
                names.append(a_name)
            else:
                # TODO: what if they both have np.nan for their names?
                names.append(None)
        return names

    def _wrap_intersection_result(self, other, result) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: finishes an intersection. Parameters: other and result (the computed
        # intersection). Obtains the consensus names from the setop conversion and sets
        # them on the result, which is returned.
        _, result_names = self._convert_can_do_setop(other)
        return result.set_names(result_names)

    def _wrap_difference_result(self, other, result: MultiIndex) -> MultiIndex:
        # [FUNCTION DOC]
        # HELPER: finishes a difference. Parameters: other and result (a MultiIndex).
        # Obtains the consensus names from the setop conversion; if the result is empty
        # its unused levels are removed first. Returns the result with those names set.
        _, result_names = self._convert_can_do_setop(other)

        if len(result) == 0:
            return result.remove_unused_levels().set_names(result_names)
        else:
            return result.set_names(result_names)

    def _convert_can_do_setop(self, other):
        # [FUNCTION DOC]
        # HELPER: prepares the other operand of a set operation. Parameter other is an
        # index or list-like of tuples. Returns a pair of the other operand as an index and
        # the result names. A non-index empty list-like returns an empty slice of this index
        # with this index's names. A non-empty non-index is converted from tuples with this
        # index's names, and a value or type error is converted to a type error stating
        # that other must be a MultiIndex or a list of tuples. An index operand gives the
        # common names of both (names that agree, else none).
        result_names = self.names

        if not isinstance(other, Index):
            if len(other) == 0:
                return self[:0], self.names
            else:
                msg = "other must be a MultiIndex or a list of tuples"
                try:
                    other = MultiIndex.from_tuples(other, names=self.names)
                except (ValueError, TypeError) as err:
                    # ValueError raised by tuples_to_object_array if we
                    #  have non-object dtype
                    raise TypeError(msg) from err
        else:
            result_names = get_unanimous_names(self, other)

        return other, result_names

    # --------------------------------------------------------------------

    def astype(self, dtype, copy: bool = True):
        # [FUNCTION DOC]
        # Cast the values; only the object dtype is supported. Parameters: dtype (numpy
        # or pandas dtype, resolved) and copy (default true). Raises a not-implemented
        # error for categorical dtypes and a type error for every other non-object
        # dtype. For object dtype returns a view when copy is true, otherwise self.
        # KNOWN ISSUE: the docstring describes integer-dtype normalization and speaks of conversion to other index classes, none of which applies, since any dtype other than object is rejected.
        """
        Create an MultiIndex with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a TypeError exception is raised.

        Parameters
        ----------
        dtype : numpy dtype or pandas type
            Note that any signed integer `dtype` is treated as ``'int64'``,
            and any unsigned integer `dtype` is treated as ``'uint64'``,
            regardless of the size.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        MultiIndex
            MultiIndex with values cast to specified dtype.

        See Also
        --------
        Index.dtype: Return the dtype object of the underlying data.
        Index.dtypes: Return the dtype object of the underlying data.
        Index.convert_dtypes: Convert columns to the best possible dtypes.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(([1, 2, 3], [4, 5, 6]))
        >>> mi
        MultiIndex([(1, 4),
                    (2, 5),
                    (3, 6)],
                   )
        >>> mi.astype("object")
        MultiIndex([(1, 4),
                    (2, 5),
                    (3, 6)],
                   )
        """
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, CategoricalDtype):
            msg = "> 1 ndim Categorical are not supported at this time"
            raise NotImplementedError(msg)
        if not is_object_dtype(dtype):
            raise TypeError(
                "Setting a MultiIndex dtype to anything other than object "
                "is not supported"
            )
        if copy is True:
            return self._view()
        return self

    def _validate_fill_value(self, item):
        # [FUNCTION DOC]
        # HELPER: validates an item to insert. Parameter item: a MultiIndex (must have the
        # same number of levels, else value error; its tuple values are returned), a
        # non-tuple (padded with empty strings for the missing lower levels), or a tuple
        # (must have exactly one element per level, else value error). Returns the item
        # as a tuple (or the values array for a MultiIndex).
        if isinstance(item, MultiIndex):
            # GH#43212
            if item.nlevels != self.nlevels:
                raise ValueError("Item must have length equal to number of levels.")
            return item._values
        elif not isinstance(item, tuple):
            # Pad the key with empty strings if lower levels of the key
            # aren't specified:
            item = (item,) + ("",) * (self.nlevels - 1)
        elif len(item) != self.nlevels:
            raise ValueError("Item must have length equal to number of levels.")
        return item

    def putmask(self, mask, value: MultiIndex) -> MultiIndex:
        # [FUNCTION DOC]
        # Return a new MultiIndex with the masked positions replaced. Parameters: mask
        # (boolean array-like) and value (a MultiIndex, either of the same length as this
        # index or of length one). If the mask selects nothing, a copy is returned.
        # Otherwise the replacement values (selected by mask when their length matches the
        # index) are reduced to used levels, each level is extended with the replacement
        # level's labels (union without sorting), and the codes of the masked positions
        # are overwritten with the codes of the replacements. Returns the new index
        # without verification.
        """
        Return a new MultiIndex of the values set with the mask.

        Parameters
        ----------
        mask : array like
        value : MultiIndex
            Must either be the same length as self or length one

        Returns
        -------
        MultiIndex
        """
        mask, noop = validate_putmask(self, mask)
        if noop:
            return self.copy()

        if len(mask) == len(value):
            subset = value[mask].remove_unused_levels()
        else:
            subset = value.remove_unused_levels()

        new_levels = []
        new_codes = []

        for i, (value_level, level, level_codes) in enumerate(
            zip(subset.levels, self.levels, self.codes, strict=True)
        ):
            new_level = level.union(value_level, sort=False)
            value_codes = new_level.get_indexer_for(subset.get_level_values(i))
            new_code = ensure_int64(level_codes)
            new_code[mask] = value_codes
            new_levels.append(new_level)
            new_codes.append(new_code)

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False
        )

    def insert(self, loc: int, item) -> MultiIndex:
        # [FUNCTION DOC]
        # Insert a row. Parameters: loc (integer position) and item (a tuple with one
        # label per level; a non-tuple is padded with empty strings; a MultiIndex is also
        # accepted). Raises a value error when the item has the wrong number of levels.
        # Labels that are not in a level are appended to that level (a missing-value label
        # gets the missing code, GH 59003 and 60388). Returns a new MultiIndex with the
        # codes inserted at loc, without verification.
        """
        Make new MultiIndex inserting new item at location

        Parameters
        ----------
        loc : int
        item : tuple
            Must be same length as number of levels in the MultiIndex

        Returns
        -------
        new_index : Index
        """
        item = self._validate_fill_value(item)

        new_levels = []
        new_codes = []
        for k, level, level_codes in zip(item, self.levels, self.codes, strict=True):
            if k not in level:
                # have to insert into level
                # must insert at end otherwise you have to recompute all the
                # other codes
                lev_loc = len(level)
                level = level.insert(lev_loc, k)
                if isna(level[lev_loc]):  # GH 59003, 60388
                    lev_loc = -1
            else:
                lev_loc = level.get_loc(k)

            new_levels.append(level)
            new_codes.append(np.insert(ensure_int64(level_codes), loc, lev_loc))

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False
        )

    def delete(self, loc) -> MultiIndex:
        # [FUNCTION DOC]
        # Delete rows. Parameter loc is an integer position or array of positions. The
        # code arrays are deleted at those positions (numpy rules apply and can raise an
        # index error). Returns a new MultiIndex that keeps the levels and names, without
        # verification and without sortorder.
        """
        Make new index with passed location deleted

        Returns
        -------
        new_index : MultiIndex
        """
        new_codes = [np.delete(level_codes, loc) for level_codes in self.codes]
        return MultiIndex(
            levels=self.levels,
            codes=new_codes,
            names=self.names,
            verify_integrity=False,
        )

    def isin(self, values, level=None) -> npt.NDArray[np.bool_]:
        # [FUNCTION DOC]
        # Test membership. Parameters: values (set or list-like; generators are consumed;
        # tuples when no level is given) and level (default none; a name or number of the
        # level to test). Without level: an empty collection gives all false; otherwise the
        # values are converted to a MultiIndex (raises on non-tuples) and each row is
        # looked up among the unique values. With level: the labels of that level are
        # tested with the level's own membership test (all false if empty). Returns a
        # boolean array.
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array of whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Parameters
        ----------
        values : set or list-like
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index is a
            `MultiIndex`).

        Returns
        -------
        np.ndarray[bool]
            NumPy array of boolean values.

        See Also
        --------
        Series.isin : Same for Series.
        DataFrame.isin : Same method for DataFrames.

        Notes
        -----
        In the case of `MultiIndex` you must either specify `values` as a
        list-like object containing tuples that are the same length as the
        number of levels, or specify `level`. Otherwise it will raise a
        ``ValueError``.

        If `level` is specified:

        - if it is the name of one *and only one* index level, use that level;
        - otherwise it should be a number indicating level position.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])

        >>> mi = pd.MultiIndex.from_arrays(
        ...     [[1, 2, 3], ["red", "blue", "green"]], names=["number", "color"]
        ... )
        >>> mi
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> mi.isin(["red", "orange", "yellow"], level="color")
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> mi.isin([(1, "red"), (3, "red")])
        array([ True, False, False])
        """
        if isinstance(values, Generator):
            values = list(values)

        if level is None:
            if len(values) == 0:
                return np.zeros((len(self),), dtype=np.bool_)
            if not isinstance(values, MultiIndex):
                values = MultiIndex.from_tuples(values)
            return values.unique().get_indexer_for(self) != -1
        else:
            num = self._get_level_number(level)
            levs = self.get_level_values(num)

            if levs.size == 0:
                return np.zeros(len(levs), dtype=np.bool_)
            return levs.isin(values)

    # error: Incompatible types in assignment (expression has type overloaded function,
    # base class "Index" defined the type as "Callable[[Index, Any, bool], Any]")
    rename = Index.set_names  # type: ignore[assignment]

    # ---------------------------------------------------------------
    # Arithmetic/Numeric Methods - Disabled

    __add__ = make_invalid_op("__add__")
    __radd__ = make_invalid_op("__radd__")
    __iadd__ = make_invalid_op("__iadd__")
    __sub__ = make_invalid_op("__sub__")
    __rsub__ = make_invalid_op("__rsub__")
    __isub__ = make_invalid_op("__isub__")
    __pow__ = make_invalid_op("__pow__")
    __rpow__ = make_invalid_op("__rpow__")
    __mul__ = make_invalid_op("__mul__")
    __rmul__ = make_invalid_op("__rmul__")
    __floordiv__ = make_invalid_op("__floordiv__")
    __rfloordiv__ = make_invalid_op("__rfloordiv__")
    __truediv__ = make_invalid_op("__truediv__")
    __rtruediv__ = make_invalid_op("__rtruediv__")
    __mod__ = make_invalid_op("__mod__")
    __rmod__ = make_invalid_op("__rmod__")
    __divmod__ = make_invalid_op("__divmod__")
    __rdivmod__ = make_invalid_op("__rdivmod__")
    # Unary methods disabled
    __neg__ = make_invalid_op("__neg__")
    __pos__ = make_invalid_op("__pos__")
    __abs__ = make_invalid_op("__abs__")
    __invert__ = make_invalid_op("__invert__")


def _lexsort_depth(codes: list[np.ndarray], nlevels: int) -> int:
    # [FUNCTION DOC]
    # HELPER: lexsort depth of code arrays. Parameters: codes (list of integer arrays)
    # and nlevels. Tries prefixes of decreasing length from nlevels down to one and
    # returns the largest number of leading levels whose codes are lexicographically
    # sorted, or zero.
    """Count depth (up to a maximum of `nlevels`) with which codes are lexsorted."""
    int64_codes = [ensure_int64(level_codes) for level_codes in codes]
    for k in range(nlevels, 0, -1):
        if libalgos.is_lexsorted(int64_codes[:k]):
            return k
    return 0


def sparsify_labels(label_list, start: int = 0, sentinel: object = ""):
    # [FUNCTION DOC]
    # HELPER: sparsifies rows of label lists for display. Parameters: label_list (one
    # list of strings per level), start (default 0: index of the first row that is kept
    # fully) and sentinel (default empty string: the value used to replace repeated
    # leading labels). The labels are transposed to rows; for each row after the
    # start the leading entries equal to the previous row are replaced by the
    # sentinel until the first difference, after which the row is kept as is.
    # Returns the result transposed back to a list of per-level tuples.
    pivoted = list(zip(*label_list, strict=True))
    k = len(label_list)

    result = pivoted[: start + 1]
    prev = pivoted[start]

    for cur in pivoted[start + 1 :]:
        sparse_cur = []

        for i, (p, t) in enumerate(zip(prev, cur, strict=True)):
            if i == k - 1:
                sparse_cur.append(t)
                result.append(sparse_cur)  # type: ignore[arg-type]
                break

            if p == t:
                sparse_cur.append(sentinel)
            else:
                sparse_cur.extend(cur[i:])
                result.append(sparse_cur)  # type: ignore[arg-type]
                break

        prev = cur

    return list(zip(*result, strict=True))


def _get_na_rep(dtype: DtypeObj) -> str:
    # [FUNCTION DOC]
    # HELPER: text for missing values by dtype. Parameter dtype. For extension dtypes
    # the dtype's own missing value in text form; datetime and timedelta numpy
    # types give NaT; everything else gives NaN.
    if isinstance(dtype, ExtensionDtype):
        return f"{dtype.na_value}"
    else:
        dtype_type = dtype.type

    return {np.datetime64: "NaT", np.timedelta64: "NaT"}.get(dtype_type, "NaN")


def maybe_droplevels(index: Index, key) -> Index:
    # [FUNCTION DOC]
    # Drop leading levels of an index after a key lookup. Parameters: index (an Index
    # or MultiIndex) and key (a scalar or tuple; the caller guarantees the key is not
    # an entry of the first level). For a tuple key, one level is dropped per key
    # element, and if that would drop too much (a value error) the original index is
    # returned. For a scalar key, the first level is dropped if possible, else the
    # index is returned unchanged. Returns the resulting index.
    """
    Attempt to drop level or levels from the given index.

    Parameters
    ----------
    index: Index
    key : scalar or tuple

    Returns
    -------
    Index
    """
    # drop levels
    original_index = index
    if isinstance(key, tuple):
        # Caller is responsible for ensuring the key is not an entry in the first
        #  level of the MultiIndex.
        for _ in key:
            try:
                index = index._drop_level_numbers([0])
            except ValueError:
                # we have dropped too much, so back out
                return original_index
    else:
        try:
            index = index._drop_level_numbers([0])
        except ValueError:
            pass

    return index


def _coerce_indexer_frozen(array_like, categories, copy: bool = False) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: coerces a code array to the smallest integer dtype able to hold the
    # categories. Parameters: array_like (codes), categories (the level) and copy
    # (default false: copy the array). Returns a read-only numpy array. Note the
    # read-only flag is set on the returned array, which is the input itself when
    # no copy is made and no cast is needed.
    """
    Coerce the array-like indexer to the smallest integer dtype that can encode all
    of the given categories.

    Parameters
    ----------
    array_like : array-like
    categories : array-like
    copy : bool

    Returns
    -------
    np.ndarray
        Non-writeable.
    """
    array_like = coerce_indexer_dtype(array_like, categories)
    if copy:
        array_like = array_like.copy()
    array_like.flags.writeable = False
    return array_like


def _require_listlike(level, arr, arrname: str):
    # [FUNCTION DOC]
    # HELPER: validates arguments for the level and codes setters. Parameters: level
    # (none, a scalar, or list-like), arr (the new levels or codes) and arrname (text
    # for error messages, such as Levels or Codes). A scalar level requires arr to be
    # a list-like of non-list-likes (else type error) and both get wrapped in lists.
    # A none or list-like level requires arr to be a list of list-likes (else type
    # error). Returns the pair (level, arr).
    """
    Ensure that level is either None or listlike, and arr is list-of-listlike.
    """
    if level is not None and not is_list_like(level):
        if not is_list_like(arr):
            raise TypeError(f"{arrname} must be list-like")
        if len(arr) > 0 and is_list_like(arr[0]):
            raise TypeError(f"{arrname} must be list-like")
        level = [level]
        arr = [arr]
    elif level is None or is_list_like(level):
        if not is_list_like(arr) or not is_list_like(arr[0]):
            raise TypeError(f"{arrname} must be list of lists-like")
    return level, arr


def cartesian_product(X: list[np.ndarray]) -> list[np.ndarray]:
    # [FUNCTION DOC]
    # Cartesian product of arrays as parallel arrays (numpy analogue of the standard
    # product). Parameter X is a list-like of list-likes (anything else raises a
    # type error). Returns a list with one array per input, each of length equal to
    # the product of the input lengths, where earlier inputs vary slowest; an empty
    # input list returns an empty list, and an empty factor gives empty arrays.
    # Raises a value error if the product size overflows.
    """
    Numpy version of itertools.product.
    Sometimes faster (for large inputs)...

    Parameters
    ----------
    X : list-like of list-likes

    Returns
    -------
    product : list of ndarrays

    Examples
    --------
    >>> cartesian_product([list("ABC"), [1, 2]])
    [array(['A', 'A', 'B', 'B', 'C', 'C'], dtype='<U1'), array([1, 2, 1, 2, 1, 2])]

    See Also
    --------
    itertools.product : Cartesian product of input iterables.  Equivalent to
        nested for-loops.
    """
    msg = "Input must be a list-like of list-likes"
    if not is_list_like(X):
        raise TypeError(msg)
    for x in X:
        if not is_list_like(x):
            raise TypeError(msg)

    if len(X) == 0:
        return []

    lenX = np.fromiter((len(x) for x in X), dtype=np.intp)
    cumprodX = np.cumprod(lenX)

    if np.any(cumprodX < 0):
        raise ValueError("Product space too large to allocate arrays!")

    a = np.roll(cumprodX, 1)
    a[0] = 1

    if cumprodX[-1] != 0:
        b = cumprodX[-1] / cumprodX
    else:
        # if any factor is empty, the cartesian product is empty
        b = np.zeros_like(cumprodX)

    return [
        np.tile(
            np.repeat(x, b[i]),
            np.prod(a[i]),
        )
        for i, x in enumerate(X)
    ]
