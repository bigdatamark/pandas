"""
Utility functions related to concat.
"""

# [MODULE DOC]
# Concatenation helpers for arrays of possibly different dtypes. Public contents:
# concat_compat (concatenate a sequence of numpy or extension arrays with dtype
# reconciliation) and union_categoricals (exposed in the public types namespace; combine
# categorical-like inputs by unioning categories). Private: a helper that filters
# empties and one that decides the result dtype.
# Relationships: used by the block manager, Index and concat machinery; relies on the
# dtype casting helpers in the cast and astype modules, on the categorical dtype, and on
# the generic ABC classes. Categorical-specific functions are imported lazily inside
# union_categoricals to avoid circular imports.
# Import-time side effects: none.


from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import lib
from pandas.util._decorators import set_module

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
    common_dtype_categorical_compat,
    find_common_type,
    np_find_common_type,
)
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
    )

    from pandas.core.arrays import (
        Categorical,
        ExtensionArray,
    )


def _is_nonempty(x: ArrayLike, axis: AxisInt) -> bool:
    # [FUNCTION DOC]
    # HELPER: for concat_compat. Parameters: x, an array; axis, the concatenation axis.
    # Returns true if x has fewer dimensions than needed for the axis to exist
    # (one-dimensional arrays are always kept), otherwise whether the length along that
    # axis is greater than zero.
    # filter empty arrays
    # 1-d dtypes always are included here
    if x.ndim <= axis:
        return True
    return x.shape[axis] > 0


def concat_compat(
    to_concat: Sequence[ArrayLike], axis: AxisInt = 0, ea_compat_axis: bool = False
) -> ArrayLike:
    # [FUNCTION DOC]
    # Concatenates a sequence of arrays and returns a single array, trying to preserve
    # the combined dtype.
    # Parameters: to_concat, a sequence of numpy arrays or extension arrays; axis
    # (default 0), the axis along which to join; ea_compat_axis (default false), when
    # true extension arrays are joined as if they were one-dimensional and the axis
    # keyword is never passed to them.
    # Behavior: (1) Fast path when the sequence is non-empty and all dtypes are equal:
    # numpy arrays go to the numpy concatenation; extension arrays use their same-type
    # concatenation, passing the axis only when it is nonzero and ea_compat_axis is
    # false. (2) Otherwise empty arrays along the axis are identified, and the result
    # dtype is determined by the private result-dtype helper; if a target dtype was
    # determined every input is cast to it without copying when possible. (3) If the
    # first array is then an extension array, its class concatenates all (axis passed
    # only when nonzero and not ea_compat_axis). (4) Otherwise numpy concatenates; if no
    # extension arrays were present, boolean kinds were among the inputs and the result
    # came out as integer, unsigned or float, the result is converted to object dtype to
    # avoid turning booleans into numbers.
    # Raises whatever the underlying concatenation raises (for example on an empty
    # sequence in the slow path, indexing the first element fails with an index error).
    # No inputs are mutated.
    """
    provide concatenation of an array of arrays each of which is a single
    'normalized' dtypes (in that for example, if it's object, then it is a
    non-datetimelike and provide a combined dtype for the resulting array that
    preserves the overall dtype if possible)

    Parameters
    ----------
    to_concat : sequence of arrays
    axis : axis to provide concatenation
    ea_compat_axis : bool, default False
        For ExtensionArray compat, behave as if axis == 1 when determining
        whether to drop empty arrays.

    Returns
    -------
    a single array, preserving the combined dtypes
    """
    if len(to_concat) and lib.dtypes_all_equal([obj.dtype for obj in to_concat]):
        # fastpath!
        obj = to_concat[0]
        if isinstance(obj, np.ndarray):
            to_concat_arrs = cast("Sequence[np.ndarray]", to_concat)
            return np.concatenate(to_concat_arrs, axis=axis)

        to_concat_eas = cast("Sequence[ExtensionArray]", to_concat)
        if ea_compat_axis:
            # We have 1D objects, that don't support axis keyword
            return obj._concat_same_type(to_concat_eas)
        elif axis == 0:
            return obj._concat_same_type(to_concat_eas)
        else:
            # e.g. DatetimeArray
            # NB: We are assuming here that ensure_wrapped_if_arraylike has
            #  been called where relevant.
            return obj._concat_same_type(
                # error: Unexpected keyword argument "axis" for "_concat_same_type"
                # of "ExtensionArray"
                to_concat_eas,
                axis=axis,  # type: ignore[call-arg]
            )

    # If all arrays are empty, there's nothing to convert, just short-cut to
    # the concatenation, #3121.
    #
    # Creating an empty array directly is tempting, but the winnings would be
    # marginal given that it would still require shape & dtype calculation and
    # np.concatenate which has them both implemented is compiled.
    non_empties = [x for x in to_concat if _is_nonempty(x, axis)]

    any_ea, kinds, target_dtype = _get_result_dtype(to_concat, non_empties)

    if target_dtype is not None:
        to_concat = [astype_array(arr, target_dtype, copy=False) for arr in to_concat]

    if not isinstance(to_concat[0], np.ndarray):
        # i.e. isinstance(to_concat[0], ExtensionArray)
        to_concat_eas = cast("Sequence[ExtensionArray]", to_concat)
        cls = type(to_concat[0])
        # GH#53640: eg. for datetime array, axis=1 but 0 is default
        # However, class method `_concat_same_type()` for some classes
        # may not support the `axis` keyword
        if ea_compat_axis or axis == 0:
            return cls._concat_same_type(to_concat_eas)
        else:
            return cls._concat_same_type(
                to_concat_eas,
                axis=axis,  # type: ignore[call-arg]
            )
    else:
        to_concat_arrs = cast("Sequence[np.ndarray]", to_concat)
        result = np.concatenate(to_concat_arrs, axis=axis)

        if not any_ea and "b" in kinds and result.dtype.kind in "iuf":
            # GH#39817 cast to object instead of casting bools to numeric
            result = result.astype(object, copy=False)
    return result


def _get_result_dtype(
    to_concat: Sequence[ArrayLike], non_empties: Sequence[ArrayLike]
) -> tuple[bool, set[str], DtypeObj | None]:
    # [FUNCTION DOC]
    # HELPER: for concat_compat. Parameters: to_concat, all arrays; non_empties, those
    # that are non-empty along the axis. Returns a 3-tuple: whether any input is an
    # extension array, the set of numpy kind codes (possibly replaced by the object code
    # when coerced to object), and the target dtype or none when no cast is needed.
    # Rules: if any extension arrays are present and dtypes differ, the target is the
    # common type of all dtypes, adjusted for categorical compatibility. If all inputs
    # are numpy and all are empty with differing kinds, the target is object unless the
    # kinds are all numeric (integer, unsigned, float) or all among boolean, integer,
    # unsigned, in which case numpy is left to decide. If non-empty inputs mix booleans
    # with other kinds, the target is object. Otherwise the target is the numpy common
    # type of the distinct dtypes.
    target_dtype = None

    dtypes = {obj.dtype for obj in to_concat}
    kinds = {obj.dtype.kind for obj in to_concat}

    any_ea = any(not isinstance(x, np.ndarray) for x in to_concat)
    if any_ea:
        # i.e. any ExtensionArrays

        # we ignore axis here, as internally concatting with EAs is always
        # for axis=0
        if len(dtypes) != 1:
            target_dtype = find_common_type([x.dtype for x in to_concat])
            target_dtype = common_dtype_categorical_compat(to_concat, target_dtype)

    elif not len(non_empties):
        # we have all empties, but may need to coerce the result dtype to
        # object if we have non-numeric type operands (numpy would otherwise
        # cast this to float)
        if len(kinds) != 1:
            if not len(kinds - {"i", "u", "f"}) or not len(kinds - {"b", "i", "u"}):
                # let numpy coerce
                pass
            else:
                # coerce to object
                target_dtype = np.dtype(object)
                kinds = {"o"}
    elif "b" in kinds and len(kinds) > 1:
        # GH#21108, GH#45101
        target_dtype = np.dtype(object)
        kinds = {"o"}
    else:
        # error: Argument 1 to "np_find_common_type" has incompatible type
        # "*Set[Union[ExtensionDtype, Any]]"; expected "dtype[Any]"
        target_dtype = np_find_common_type(*dtypes)  # type: ignore[arg-type]

    return any_ea, kinds, target_dtype


@set_module("pandas.api.types")
def union_categoricals(
    to_union, sort_categories: bool = False, ignore_order: bool = False
) -> Categorical:
    # [FUNCTION DOC]
    # Combines a list-like of categorical-like objects (Categorical, CategoricalIndex or
    # categorical Series) into a single plain Categorical whose categories are the
    # union.
    # Parameters: to_union, the list-like of inputs; sort_categories (default false),
    # sort the resulting categories (not allowed with ordered inputs unless order is
    # ignored); ignore_order (default false), treat all inputs as unordered and return
    # an unordered result.
    # Returns a new Categorical built from concatenated codes and a new categorical
    # dtype.
    # Raises a value error for an empty input; a type error if any element is not
    # categorical-like, if category dtypes differ, if sorting categories is requested
    # with ordered inputs having identical categories, if inputs are ordered with
    # different categories (unless order is ignored), or if some are ordered and some
    # not.
    # Algorithm: unwrap each input to its Categorical; if all share the same categories
    # up to permutation, re-encode every input against the first one's categories,
    # concatenate codes, and when sorting recode through the sorted categories;
    # otherwise, when order is ignored or no input is ordered, append all category
    # indexes, drop duplicates (first-appearance order), optionally sort, and recode
    # each input; otherwise raise the matching error. Ordered flag is taken from the
    # first input in the identical case and forced false when order is ignored. Inputs
    # are not mutated.
    # KNOWN ISSUE: the docstring example text mentions an option named ignore_ordered,
    # but the parameter is actually named ignore_order.
    """
    Combine list-like of Categorical-like, unioning categories.

    All categories must have the same dtype.

    Parameters
    ----------
    to_union : list-like
        Categorical, CategoricalIndex, or Series with dtype='category'.
    sort_categories : bool, default False
        If true, resulting categories will be lexsorted, otherwise
        they will be ordered as they appear in the data.
    ignore_order : bool, default False
        If true, the ordered attribute of the Categoricals will be ignored.
        Results in an unordered categorical.

    Returns
    -------
    Categorical
        The union of categories being combined.

    Raises
    ------
    TypeError
        - all inputs do not have the same dtype
        - all inputs do not have the same ordered property
        - all inputs are ordered and their categories are not identical
        - sort_categories=True and Categoricals are ordered
    ValueError
        Empty list of categoricals passed

    See Also
    --------
    CategoricalDtype : Type for categorical data with the categories and orderedness.
    Categorical : Represent a categorical variable in classic R / S-plus fashion.

    Notes
    -----
    To learn more about categories, see `link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#unioning>`__

    Examples
    --------
    If you want to combine categoricals that do not necessarily have
    the same categories, `union_categoricals` will combine a list-like
    of categoricals. The new categories will be the union of the
    categories being combined.

    >>> a = pd.Categorical(["b", "c"])
    >>> b = pd.Categorical(["a", "b"])
    >>> pd.api.types.union_categoricals([a, b])
    ['b', 'c', 'a', 'b']
    Categories (3, str): ['b', 'c', 'a']

    By default, the resulting categories will be ordered as they appear
    in the `categories` of the data. If you want the categories to be
    lexsorted, use `sort_categories=True` argument.

    >>> pd.api.types.union_categoricals([a, b], sort_categories=True)
    ['b', 'c', 'a', 'b']
    Categories (3, str): ['a', 'b', 'c']

    `union_categoricals` also works with the case of combining two
    categoricals of the same categories and order information (e.g. what
    you could also `append` for).

    >>> a = pd.Categorical(["a", "b"], ordered=True)
    >>> b = pd.Categorical(["a", "b", "a"], ordered=True)
    >>> pd.api.types.union_categoricals([a, b])
    ['a', 'b', 'a', 'b', 'a']
    Categories (2, str): ['a' < 'b']

    Raises `TypeError` because the categories are ordered and not identical.

    >>> a = pd.Categorical(["a", "b"], ordered=True)
    >>> b = pd.Categorical(["a", "b", "c"], ordered=True)
    >>> pd.api.types.union_categoricals([a, b])
    Traceback (most recent call last):
        ...
    TypeError: to union ordered Categoricals, all categories must be the same

    Ordered categoricals with different categories or orderings can be
    combined by using the `ignore_ordered=True` argument.

    >>> a = pd.Categorical(["a", "b", "c"], ordered=True)
    >>> b = pd.Categorical(["c", "b", "a"], ordered=True)
    >>> pd.api.types.union_categoricals([a, b], ignore_order=True)
    ['a', 'b', 'c', 'c', 'b', 'a']
    Categories (3, str): ['a', 'b', 'c']

    `union_categoricals` also works with a `CategoricalIndex`, or `Series`
    containing categorical data, but note that the resulting array will
    always be a plain `Categorical`

    >>> a = pd.Series(["b", "c"], dtype="category")
    >>> b = pd.Series(["a", "b"], dtype="category")
    >>> pd.api.types.union_categoricals([a, b])
    ['b', 'c', 'a', 'b']
    Categories (3, str): ['b', 'c', 'a']
    """
    from pandas import Categorical
    from pandas.core.arrays.categorical import recode_for_categories

    if len(to_union) == 0:
        raise ValueError("No Categoricals to union")

    def _maybe_unwrap(x):
        # [FUNCTION DOC]
        # HELPER: for union_categoricals. Returns the underlying Categorical for a
        # CategoricalIndex or Series (their internal values), returns a Categorical
        # unchanged, and raises a type error for anything else.
        if isinstance(x, (ABCCategoricalIndex, ABCSeries)):
            return x._values
        elif isinstance(x, Categorical):
            return x
        else:
            raise TypeError("all components to combine must be Categorical")

    to_union = [_maybe_unwrap(x) for x in to_union]
    first = to_union[0]

    if not lib.dtypes_all_equal([obj.categories.dtype for obj in to_union]):
        raise TypeError("dtype of categories must be the same")

    ordered = False
    if all(first._categories_match_up_to_permutation(other) for other in to_union[1:]):
        # identical categories - fastpath
        categories = first.categories
        ordered = first.ordered

        all_codes = [first._encode_with_my_categories(x)._codes for x in to_union]
        new_codes = np.concatenate(all_codes)

        if sort_categories and not ignore_order and ordered:
            raise TypeError("Cannot use sort_categories=True with ordered Categoricals")

        if sort_categories and not categories.is_monotonic_increasing:
            categories = categories.sort_values()
            indexer = categories.get_indexer(first.categories)

            from pandas.core.algorithms import take_nd

            new_codes = take_nd(indexer, new_codes, fill_value=-1)
    elif ignore_order or all(not c.ordered for c in to_union):
        # different categories - union and recode
        cats = first.categories.append([c.categories for c in to_union[1:]])
        categories = cats.unique()
        if sort_categories:
            categories = categories.sort_values()

        all_codes = [
            recode_for_categories(c.codes, c.categories, categories, copy=False)
            for c in to_union
        ]
        new_codes = np.concatenate(all_codes)
    else:
        # ordered - to show a proper error message
        if all(c.ordered for c in to_union):
            msg = "to union ordered Categoricals, all categories must be the same"
            raise TypeError(msg)
        raise TypeError("Categorical.ordered must be the same")

    if ignore_order:
        ordered = False

    dtype = CategoricalDtype(categories=categories, ordered=ordered)
    return Categorical._simple_new(new_codes, dtype=dtype)
