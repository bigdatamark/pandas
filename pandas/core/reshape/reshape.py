# [MODULE DOC]
# Stack and unstack machinery: converting between wide and long layouts by moving
# index levels to columns and the reverse.
# Public-style contents: the unstack function (dispatches for Series and DataFrame),
# the stack function and its variants (stack for the legacy implementation, stack_multiple
# for several levels, stack_v3 for the newer implementation), and stack_reshape, the
# data-moving core of the newer implementation. Private pieces: the unstacker helper
# class doing the index arithmetic, multi-level unstack, extension-array Series
# unstack, and multi-column stack helpers.
# Relationships: these are called by the stack/unstack methods of DataFrame and Series,
# and by pivot code; low-level loops live in the compiled reshape extension module;
# grouping and sorting utilities come from the sorting module.
# No import-time side effects beyond imports.
from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config.config import get_option

import pandas._libs.reshape as libreshape
from pandas.errors import (
    Pandas4Warning,
    PerformanceWarning,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_integer,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

import pandas.core.algorithms as algos
from pandas.core.algorithms import (
    factorize,
    unique,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
)
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_compressed_ids,
    get_group_index,
    get_group_index_sorter,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Level,
        npt,
    )

    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.frozen import FrozenList


class _Unstacker:
    # [CLASS DOC]
    # HELPER: for unstack and pivot; performs the index bookkeeping needed to move one
    # level of a hierarchical row index into columns.
    # State: constructor (frame type used for results), sort (whether result labels are
    # sorted), index (the hierarchical index with unused levels removed), level (integer
    # position of the level being moved), has_nan and lift (whether the moved level has
    # missing labels and the offset used to give them position zero when sorting),
    # new_index_levels, new_index_names (the remaining levels and names), removed_name,
    # removed_level, removed_level_full (the moved level and its full, unreduced
    # version), unique_nan_index (position of the missing label in unsorted mode),
    # full_shape (number of output rows and the stride of the moved level), group_index,
    # mask (which cells of the output grid are populated) and compressor (positions
    # of the first occurrence of each output row).
    # Invariants: the mask must cover at least as many cells as index entries, otherwise
    # construction fails with a value error about duplicate entries.
    # Construction may emit a performance warning if the output grid would exceed the
    # largest 32-bit integer and the performance-warnings option is on.
    # Key methods: get_result (builds the unstacked frame), get_new_values (arranges
    # values into the output grid), get_new_columns (builds the new column labels),
    # and the new_index property (new row labels).
    # KNOWN ISSUE: the docstring says the constructor may be none, defaulting to
    # DataFrame, but the code calls the constructor directly and never substitutes a
    # default; the sort parameter is not documented; the Returns and Examples sections
    # describe the unstack operation rather than this helper class.
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame will be used.

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples(
    ...     [("one", "a"), ("one", "b"), ("two", "a"), ("two", "b")]
    ... )
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
    """

    def __init__(
        self, index: MultiIndex, level: Level, constructor, sort: bool = True
    ) -> None:
        # [FUNCTION DOC]
        # Initializes the unstacker.
        # Inputs: index (hierarchical row index), level (number or name of the level to
        # move to columns), constructor (callable creating the result frame), sort
        # (default true; whether to sort labels, otherwise keeps order of appearance).
        # Effects: removes unused levels, resolves the level number, detects missing
        # labels in that level, splits the remaining levels and names from the removed
        # one, in unsorted mode reduces the removed level to the labels actually used
        # (dropping missing codes and remembering where the missing label was), warns
        # (performance warning) when the number of output cells would overflow a 32-bit
        # integer, and finally builds the selectors, which can raise a value error for
        # duplicate entries.
        self.constructor = constructor
        self.sort = sort

        self.index = index.remove_unused_levels()

        self.level = self.index._get_level_number(level)

        # `nan` values have code `-1`, when sorting, we lift to assign them
        # at index 0
        self.has_nan = -1 in self.index.codes[self.level]
        should_lift = self.has_nan and self.sort
        self.lift = 1 if should_lift else 0

        # Note: the "pop" below alters these in-place.
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)

        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]
        self.unique_nan_index: int = -1
        if not self.sort:
            unique_codes: np.ndarray = unique(self.index.codes[self.level])
            if self.has_nan:
                # drop nan codes, because they are not represented in level
                nan_mask = unique_codes == -1

                unique_codes = unique_codes[~nan_mask]
                self.unique_nan_index = np.flatnonzero(nan_mask)[0]

            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)

        if get_option("performance_warnings"):
            # Bug fix GH 20601
            # If the data frame is too big, the number of unique index combination
            # will cause int32 overflow on windows environments.
            # We want to check and raise a warning before this happens
            num_rows = max(index_level.size for index_level in self.new_index_levels)
            num_columns = self.removed_level.size

            # GH20601: This forces an overflow if the number of cells is too high.
            # GH 26314: Previous ValueError raised was too restrictive for many users.
            num_cells = num_rows * num_columns
            if num_cells > np.iinfo(np.int32).max:
                warnings.warn(
                    f"The following operation may generate {num_cells} cells "
                    f"in the resulting pandas object.",
                    PerformanceWarning,
                    stacklevel=find_stack_level(),
                )

        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(
        self,
    ) -> tuple[
        npt.NDArray[np.intp],
        list[np.ndarray],  # each has _some_ signed integer dtype
    ]:
        # [FUNCTION DOC]
        # HELPER: cached property; computes the permutation that sorts the entries by
        # the remaining levels first and the moved level last.
        # Returns a pair: the sorting indexer (platform integers) and the list of code
        # arrays in that order (remaining levels then moved level). In unsorted mode the
        # codes are refactorized first so they reflect order of appearance.
        v = self.level

        codes = list(self.index.codes)
        if not self.sort:
            # Create new codes considering that labels are already sorted
            codes = [factorize(code)[0] for code in codes]
        levs = list(self.index.levels)
        to_sort = codes[:v] + codes[v + 1 :] + [codes[v]]
        sizes = tuple(len(x) for x in levs[:v] + levs[v + 1 :] + [levs[v]])

        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)

        indexer = get_group_index_sorter(comp_index, ngroups)
        return indexer, to_sort

    @cache_readonly
    def sorted_labels(self) -> list[np.ndarray]:
        # [FUNCTION DOC]
        # Cached property; returns the code arrays (remaining levels, then moved level)
        # reordered by the sorting indexer when sorting is on, or unchanged otherwise.
        indexer, to_sort = self._indexer_and_to_sort
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        return to_sort

    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: reorders the rows of a values array by the cached sorting indexer and
        # returns the reordered array (a new array); the input is two-dimensional or
        # one-dimensional and is taken along the first axis.
        indexer, _ = self._indexer_and_to_sort
        sorted_values = algos.take_nd(values, indexer, axis=0)
        return sorted_values

    def _make_selectors(self) -> None:
        # [FUNCTION DOC]
        # HELPER: called at construction; computes the output grid layout.
        # Compresses the remaining labels into group ids (one per output row), sets the
        # grid shape to the number of groups by the width of the moved level (including
        # one extra slot for missing labels), builds a boolean mask of populated cells
        # from the sorted labels, and sets the group index, mask and compressor
        # (positions that map output rows back to first occurrences: by search when
        # sorted, by first-occurrence order otherwise).
        # Raises a value error when the number of populated cells is smaller than the
        # number of index entries, which means the index has duplicate entries.
        new_levels = self.new_index_levels

        # make the mask
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = tuple(len(x) for x in new_levels)

        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)

        comp_index = ensure_platform_int(comp_index)
        stride = self.index.levshape[self.level] + self.has_nan
        self.full_shape = ngroups, stride

        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)

        if mask.sum() < len(self.index):
            raise ValueError("Index contains duplicate entries, cannot reshape")

        self.group_index = comp_index
        self.mask = mask
        if self.sort:
            self.compressor = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    @cache_readonly
    def mask_all(self) -> bool:
        # [FUNCTION DOC]
        # Cached property; true when every cell of the output grid is populated, i.e. no
        # filling of missing combinations is required.
        return bool(self.mask.all())

    @cache_readonly
    def arange_result(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
        # [FUNCTION DOC]
        # Cached property (reused by the extension-array block unstack). Unstacks a
        # dummy array of positions of the original entries, using minus one as fill.
        # Returns the resulting position grid and a one-dimensional boolean array
        # saying, for each output column, whether it has any populated row.
        # A trailing note after the return statement is a leftover comment (no code).
        # We cache this for reuse in ExtensionBlock._unstack
        dummy_arr = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return new_values, mask.any(0)
        # TODO: in all tests we have mask.any(0).all(); can we rely on that?

    def get_result(self, obj, value_columns, fill_value) -> DataFrame:
        # [FUNCTION DOC]
        # Builds the unstacked frame.
        # Inputs: obj (Series or DataFrame whose values are unstacked), value_columns
        # (column index of obj for a frame; none for a Series), fill_value (for empty
        # cells; none selects the dtype default).
        # Takes the values (one-dimensional ones become a single column), requires
        # column labels when there is more than one column (value error otherwise,
        # marked unreachable by coverage), arranges them into the output grid, builds
        # new columns and index, and constructs the result without copying. If the result
        # shares memory with the original values (possible only when one dimension is
        # size one), registers a reference from the result to obj for copy-on-write.
        # Returns the new frame.
        values = obj._values
        if values.ndim == 1:
            values = values[:, np.newaxis]

        if value_columns is None and values.shape[1] != 1:  # pragma: no cover
            raise ValueError("must pass column labels for multi-column data")

        new_values, _ = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index

        result = self.constructor(
            new_values, index=index, columns=columns, dtype=new_values.dtype, copy=False
        )
        if isinstance(values, np.ndarray):
            base, new_base = values.base, new_values.base
        elif isinstance(values, NDArrayBackedExtensionArray):
            base, new_base = values._ndarray.base, new_values._ndarray.base
        else:
            base, new_base = 1, 2  # type: ignore[assignment]
        if base is new_base:
            # We can only get here if one of the dimensions is size 1
            result._mgr.add_references(obj._mgr)
        return result

    def get_new_values(self, values, fill_value=None):
        # [FUNCTION DOC]
        # Arranges values into the output grid of (output rows) by (columns times the
        # width of the moved level).
        # Inputs: values (array or extension array, one- or two-dimensional; one-dimensional
        # is made a single column), fill_value (for missing cells).
        # Returns a pair: the new values and a boolean mask of populated cells.
        # Fast path: when every cell is populated and values are non-empty, reshapes the
        # sorted values directly and returns an all-true mask.
        # Otherwise: for extension dtypes allocates an empty array and fills it with
        # fill_value (assuming the dtype can hold it); for other dtypes promotes the dtype
        # to hold the fill value (emitting a Pandas4Warning future-deprecation warning
        # when the promotion changes the dtype, except for the benign integer/boolean
        # to float promotion with a missing fill value), fills, converts datetime-like
        # data to 64-bit integers, then calls the compiled unstack loop to place values
        # and mark the mask; finally restores datetime-like dtype.
        # Side effects: may emit the warning above.
        if values.ndim == 1:
            values = values[:, np.newaxis]

        sorted_values = self._make_sorted_values(values)

        # place the values
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = self.mask_all

        # we can simply reshape if we don't have a mask
        if mask_all and len(values):
            # TODO: Under what circumstances can we rely on sorted_values
            #  matching values?  When that holds, we can slice instead
            #  of take (in particular for EAs)
            new_values = (
                sorted_values.reshape(length, width, stride)
                .swapaxes(1, 2)
                .reshape(result_shape)
            )
            new_mask = np.ones(result_shape, dtype=bool)
            return new_values, new_mask

        dtype = values.dtype

        if isinstance(dtype, ExtensionDtype):
            # GH#41875
            # We are assuming that fill_value can be held by this dtype,
            #  unlike the non-EA case that promotes.
            cls = dtype.construct_array_type()
            new_values = cls._empty(result_shape, dtype=dtype)
            if not mask_all:
                new_values[:] = fill_value
        else:
            if not mask_all:
                old_dtype = dtype
                dtype, fill_value = maybe_promote(dtype, fill_value)
                if old_dtype != dtype:
                    if old_dtype.kind not in "iub":
                        warnings.warn(
                            # GH#12189, GH#53868
                            "Using a fill_value that cannot be held in the existing "
                            "dtype is deprecated and will raise in a future version.",
                            Pandas4Warning,
                            stacklevel=find_stack_level(),
                        )
                    elif not isna(fill_value):
                        warnings.warn(
                            # GH#12189, GH#53868
                            "Using a fill_value that cannot be held in the existing "
                            "dtype is deprecated and will raise in a future version.",
                            Pandas4Warning,
                            stacklevel=find_stack_level(),
                        )
            new_values = np.empty(result_shape, dtype=dtype)
            if not mask_all:
                new_values.fill(fill_value)

        name = dtype.name
        new_mask = np.zeros(result_shape, dtype=bool)

        # we need to convert to a basic dtype
        # and possibly coerce an input to our output dtype
        # e.g. ints -> floats
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view("i8")
            new_values = new_values.view("i8")
        else:
            sorted_values = sorted_values.astype(name, copy=False)

        # fill in our values & mask
        libreshape.unstack(
            sorted_values,
            mask.view("u1"),
            stride,
            length,
            width,
            new_values,
            new_mask.view("u1"),
        )

        # reconstruct dtype if needed
        if needs_i8_conversion(values.dtype):
            # view as datetime64 so we can wrap in DatetimeArray and use
            #  DTA's view method
            new_values = new_values.view("M8[ns]")
            new_values = ensure_wrapped_if_datetimelike(new_values)
            new_values = new_values.view(values.dtype)

        return new_values, new_mask

    def get_new_columns(self, value_columns: Index | None):
        # [FUNCTION DOC]
        # Builds the column labels of the unstacked result.
        # Input: value_columns (the original column index, or none for a Series).
        # For none: returns the moved level itself (with a leading missing label inserted
        # when the level had missing values) renamed to the moved level's name.
        # Otherwise returns a hierarchical index whose outer level(s) are the original
        # columns (each repeated once per moved-level label) and whose innermost level is
        # the full moved level, with codes taken from the repeater; names are the original
        # column names followed by the moved level's name. Built without verification.
        if value_columns is None:
            if not self.has_nan:
                return self.removed_level._rename(name=self.removed_name)

            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)

        stride = len(self.removed_level) + self.has_nan
        width = len(value_columns)
        propagator = np.repeat(np.arange(width), stride)

        new_levels: FrozenList | list[Index]

        if isinstance(value_columns, MultiIndex):
            new_levels = value_columns.levels + (self.removed_level_full,)  # pyright: ignore[reportOperatorIssue] # noqa: RUF005
            new_names = value_columns.names + (self.removed_name,)  # noqa: RUF005

            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [
                value_columns,
                self.removed_level_full,
            ]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]

        repeater = self._repeater

        # The entire level is then just a repetition of the single chunk:
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    @cache_readonly
    def _repeater(self) -> np.ndarray:
        # [FUNCTION DOC]
        # HELPER: cached property; the codes of the moved level, for one block of
        # output columns, expressed relative to the full (unreduced) moved level.
        # If the reduced level differs in length from the full one, maps the reduced
        # labels into the full level (with a leading minus one for missing labels).
        # Otherwise uses consecutive positions shifted by the lift; in unsorted mode with
        # missing labels, sets the missing position to minus one and shifts the later
        # positions down by one. An assertion guards that the missing position was set.
        # The two indices differ only if the unstacked level had unused items:
        if len(self.removed_level_full) != len(self.removed_level):
            # In this case, we remap the new codes to the original level:
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.has_nan:
                # insert nan index at first position
                repeater = np.insert(repeater, 0, -1)
        else:
            # Otherwise, we just use each level item exactly once:
            stride = len(self.removed_level) + self.has_nan
            repeater = np.arange(stride) - self.lift
            if self.has_nan and not self.sort:
                assert self.unique_nan_index > -1, (
                    "`unique_nan_index` not properly initialized"
                )
                # assign -1 where should be nan according to the unique values.
                repeater[self.unique_nan_index] = -1
                # compensate for the removed index level
                repeater[self.unique_nan_index + 1 :] -= 1

        return repeater

    @cache_readonly
    def new_index(self) -> MultiIndex | Index:
        # [FUNCTION DOC]
        # Cached property; the row labels of the unstacked result, independent of the
        # values.
        # Takes the codes of the remaining levels (sorted order when sorting, original
        # order otherwise), selects the first occurrence of each output row, and builds
        # either a plain index (when only one level remains, adding a missing label if
        # any code is minus one) or a hierarchical index with the remaining names.
        # Does not depend on values or value_columns
        if self.sort:
            labels = self.sorted_labels[:-1]
        else:
            v = self.level
            codes = list(self.index.codes)
            labels = codes[:v] + codes[v + 1 :]
        result_codes = [lab.take(self.compressor) for lab in labels]

        # construct the new index
        if len(self.new_index_levels) == 1:
            level, level_codes = self.new_index_levels[0], result_codes[0]
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])

        return MultiIndex(
            levels=self.new_index_levels,
            codes=result_codes,
            names=self.new_index_names,
            verify_integrity=False,
        )


def _unstack_multiple(
    data: Series | DataFrame, clocs, fill_value=None, sort: bool = True
):
    # [FUNCTION DOC]
    # HELPER: for unstack; moves several row-index levels to columns at once.
    # Inputs: data (Series or DataFrame whose index is hierarchical; the caller checks
    # this), clocs (the levels to move, as numbers or names; a single tuple name that is
    # itself a level name is treated as one level), fill_value, sort.
    # Returns data unchanged if clocs is empty. Otherwise combines the moved levels'
    # codes into a single compressed placeholder level, unstacks that placeholder, and
    # then rebuilds the column index by expanding the placeholder back into the original
    # levels (names and codes recovered from the compression).
    # For a DataFrame with hierarchical columns, falls back to unstacking the levels one
    # after another, adjusting the remaining level numbers after each step; this branch
    # returns directly. Note the source states that hierarchical columns are not yet
    # handled in the main path.
    # Returns the unstacked Series or DataFrame. Does not modify data (works on a
    # shallow copy).
    if len(clocs) == 0:
        return data

    # NOTE: This doesn't deal with hierarchical columns yet

    index = data.index
    index = cast(MultiIndex, index)  # caller is responsible for checking

    # GH 19966 Make sure if MultiIndexed index has tuple name, they will be
    # recognised as a whole
    if clocs in index.names:
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]

    rlocs = [i for i in range(index.nlevels) if i not in clocs]

    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]

    shape = tuple(len(x) for x in clevels)
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)

    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)

    if not rlocs:
        # Everything is in clocs, so the dummy df has a regular index
        dummy_index = Index(obs_ids, name="__placeholder__", copy=False)
    else:
        dummy_index = MultiIndex(
            levels=[*rlevels, obs_ids],
            codes=[*rcodes, comp_ids],
            names=[*rnames, "__placeholder__"],
            verify_integrity=False,
        )

    if isinstance(data, Series):
        dummy = data.copy(deep=False)
        dummy.index = dummy_index

        unstacked = dummy.unstack("__placeholder__", fill_value=fill_value, sort=sort)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            while clocs:
                val = clocs.pop(0)
                # error: Incompatible types in assignment (expression has type
                # "DataFrame | Series", variable has type "DataFrame")
                result = result.unstack(  # type: ignore[assignment]
                    val, fill_value=fill_value, sort=sort
                )
                clocs = [v if v < val else v - 1 for v in clocs]

            return result

        # GH#42579 deep=False to avoid consolidating
        dummy_df = data.copy(deep=False)
        dummy_df.index = dummy_index

        # error: Incompatible types in assignment (expression has type "DataFrame |
        # Series", variable has type "DataFrame")
        unstacked = dummy_df.unstack(  # type: ignore[assignment]
            "__placeholder__", fill_value=fill_value, sort=sort
        )
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        assert isinstance(unstcols, MultiIndex)  # for mypy
        new_levels = [unstcols.levels[0], *clevels]
        new_names = [data.columns.name, *cnames]

        new_codes = [unstcols.codes[0]]
        new_codes.extend(rec.take(unstcols.codes[-1]) for rec in recons_codes)

    new_columns = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns

    return unstacked


@overload
def unstack(obj: Series, level, fill_value=..., sort: bool = ...) -> DataFrame:
    # [FUNCTION DOC]
    # Typing overload only: unstacking a Series gives a DataFrame. See the final
    # implementation of unstack for behavior.
    ...


@overload
def unstack(
    obj: Series | DataFrame, level, fill_value=..., sort: bool = ...
) -> Series | DataFrame:
    # [FUNCTION DOC]
    # Typing overload only: unstacking a Series or DataFrame gives a Series or
    # DataFrame. See the final implementation of unstack for behavior.
    ...


def unstack(
    obj: Series | DataFrame, level, fill_value=None, sort: bool = True
) -> Series | DataFrame:
    # [FUNCTION DOC]
    # Moves a level of the row index into the columns (pivoting long to wide).
    # Inputs: obj (Series or DataFrame); level (number or name of the level, or a
    # list or tuple of them); fill_value (for missing combinations, default none); sort
    # (default true; sort the new labels).
    # A list/tuple with several entries delegates to the multi-level helper; with a
    # single entry it is unwrapped. For a level that is not an integer and not the
    # internal placeholder name, validates it by resolving the level number (which raises
    # for unknown levels).
    # DataFrame with hierarchical index: unstacks via the frame helper. DataFrame with a
    # flat index: transposes and stacks instead. Series with a flat index: raises a value
    # error saying a hierarchical index is required. Series with an extension dtype
    # that is one-dimensional-only: uses the extension helper to preserve dtype.
    # Otherwise uses the unstacker helper class directly.
    # Returns a DataFrame (or a Series in the flat-index DataFrame case).
    if isinstance(level, (tuple, list)):
        if len(level) != 1:
            # _unstack_multiple only handles MultiIndexes,
            # and isn't needed for a single level
            return _unstack_multiple(obj, level, fill_value=fill_value, sort=sort)
        else:
            level = level[0]

    if not is_integer(level) and not level == "__placeholder__":
        # check if level is valid in case of regular index
        obj.index._get_level_number(level)

    if isinstance(obj, DataFrame):
        if isinstance(obj.index, MultiIndex):
            return _unstack_frame(obj, level, fill_value=fill_value, sort=sort)
        else:
            return obj.T.stack()
    elif not isinstance(obj.index, MultiIndex):
        # GH 36113
        # Give nicer error messages when unstack a Series whose
        # Index is not a MultiIndex.
        raise ValueError(
            f"index must be a MultiIndex to unstack, {type(obj.index)} was passed"
        )
    else:
        if is_1d_only_ea_dtype(obj.dtype):
            return _unstack_extension_series(obj, level, fill_value, sort=sort)
        unstacker = _Unstacker(
            obj.index, level=level, constructor=obj._constructor_expanddim, sort=sort
        )
        return unstacker.get_result(obj, value_columns=None, fill_value=fill_value)


def _unstack_frame(
    obj: DataFrame, level, fill_value=None, sort: bool = True
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: for unstack; unstacks a DataFrame with a hierarchical row index.
    # Inputs: obj, level, fill_value, sort. Builds the unstacker helper; when the frame
    # cannot be cheaply transposed (mixed block layout) it delegates to the block
    # manager's unstack and wraps the result; otherwise computes the result directly with
    # the original columns. Returns the new DataFrame. The caller guarantees the index is
    # hierarchical (asserted).
    assert isinstance(obj.index, MultiIndex)  # checked by caller
    unstacker = _Unstacker(
        obj.index, level=level, constructor=obj._constructor, sort=sort
    )

    if not obj._can_fast_transpose:
        mgr = obj._mgr.unstack(unstacker, fill_value=fill_value)
        return obj._constructor_from_mgr(mgr, axes=mgr.axes)
    else:
        return unstacker.get_result(
            obj, value_columns=obj.columns, fill_value=fill_value
        )


def _unstack_extension_series(
    series: Series, level, fill_value, sort: bool
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: for unstack; unstacks a Series backed by an extension array while keeping
    # its dtype, by converting to a one-column frame, unstacking that, and then dropping
    # the outermost (original column) level of the result columns without an extra copy.
    # Inputs as in the docstring (series, level, fill_value, sort). Returns a DataFrame
    # whose columns all keep the input dtype.
    """
    Unstack an ExtensionArray-backed Series.

    The ExtensionDtype is preserved.

    Parameters
    ----------
    series : Series
        A Series with an ExtensionArray for values
    level : Any
        The level name or number.
    fill_value : Any
        The user-level (not physical storage) fill value to use for
        missing values introduced by the reshape. Passed to
        ``series.values.take``.
    sort : bool
        Whether to sort the resulting MuliIndex levels

    Returns
    -------
    DataFrame
        Each column of the DataFrame will have the same dtype as
        the input Series.
    """
    # Defer to the logic in ExtensionBlock._unstack
    df = series.to_frame()
    result = df.unstack(level=level, fill_value=fill_value, sort=sort)

    # equiv: result.droplevel(level=0, axis=1)
    #  but this avoids an extra copy
    result.columns = result.columns._drop_level_numbers([0])
    # error: Incompatible return value type (got "DataFrame | Series", expected
    # "DataFrame")
    return result  # type: ignore[return-value]


def stack(
    frame: DataFrame, level=-1, dropna: bool = True, sort: bool = True
) -> Series | DataFrame:
    # [FUNCTION DOC]
    # Legacy stack implementation: moves the columns into the row index.
    # Inputs: frame; level (column level to stack, number or name, default last; only
    # meaningful for hierarchical columns); dropna (default true; drop rows whose value
    # is missing); sort (default true; only used for hierarchical columns).
    # Hierarchical columns delegate to the multi-column helper. Otherwise builds a new
    # row index of the frame's index (levels and codes repeated for each column) combined
    # with the columns as a new innermost level (factorized when columns are not unique).
    # Values are the frame data flattened row-wise: for homogeneous extension dtypes the
    # columns are concatenated and then reordered, to keep the dtype; otherwise the
    # two-dimensional values are flattened. With dropna, missing entries and their index
    # entries are removed.
    # Returns a Series of the frame's sliced type indexed by the new index (or a DataFrame
    # when delegating for hierarchical columns).
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series or DataFrame
    """

    def stack_factorize(index):
        # [FUNCTION DOC]
        # HELPER: nested in stack; returns a (levels, codes) pair for an index: the index
        # itself with consecutive positions when it is unique, otherwise its distinct
        # categories and the code of each entry.
        if index.is_unique:
            return index, np.arange(len(index))
        codes, categories = factorize_from_iterable(index)
        return categories, codes

    N, K = frame.shape

    # Will also convert negative level numbers and check if out of bounds.
    level_num = frame.columns._get_level_number(level)

    if isinstance(frame.columns, MultiIndex):
        return _stack_multi_columns(
            frame, level_num=level_num, dropna=dropna, sort=sort
        )
    elif isinstance(frame.index, MultiIndex):
        new_levels = list(frame.index.levels)
        new_codes = [lab.repeat(K) for lab in frame.index.codes]

        clev, clab = stack_factorize(frame.columns)
        new_levels.append(clev)
        new_codes.append(np.tile(clab, N).ravel())

        new_names = list(frame.index.names)
        new_names.append(frame.columns.name)
        new_index = MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )
    else:
        levels, (ilab, clab) = zip(
            *map(stack_factorize, (frame.index, frame.columns)), strict=True
        )
        codes = ilab.repeat(K), np.tile(clab, N).ravel()
        new_index = MultiIndex(
            levels=levels,
            codes=codes,
            names=[frame.index.name, frame.columns.name],
            verify_integrity=False,
        )

    new_values: ArrayLike
    if not frame.empty and frame._is_homogeneous_type:
        # For homogeneous EAs, frame._values will coerce to object. So
        # we concatenate instead.
        dtypes = list(frame.dtypes._values)
        dtype = dtypes[0]

        if isinstance(dtype, ExtensionDtype):
            arr = dtype.construct_array_type()
            new_values = arr._concat_same_type(
                [col._values for _, col in frame.items()]
            )
            new_values = _reorder_for_extension_array_stack(new_values, N, K)
        else:
            # homogeneous, non-EA
            new_values = frame._values.ravel()

    else:
        # non-homogeneous
        new_values = frame._values.ravel()

    if dropna:
        mask = notna(new_values)
        new_values = new_values[mask]
        new_index = new_index[mask]

    return frame._constructor_sliced(new_values, index=new_index)


def stack_multiple(frame: DataFrame, level, dropna: bool = True, sort: bool = True):
    # [FUNCTION DOC]
    # Stacks several column levels one after another using the legacy stack function.
    # Inputs: frame, level (list of level names or numbers), dropna, sort (as in stack).
    # If every entry is a column level name, stacks them in order by name. Otherwise, if
    # all entries are integers, converts them to non-negative numbers and stacks them in
    # order, decrementing the remaining numbers greater than the one just stacked,
    # because stacking shifts the later levels down. A mixture raises a value error.
    # Returns the final stacked Series or DataFrame.
    # If all passed levels match up to column names, no
    # ambiguity about what to do
    if all(lev in frame.columns.names for lev in level):
        result = frame
        for lev in level:
            # error: Incompatible types in assignment (expression has type
            # "Series | DataFrame", variable has type "DataFrame")
            result = stack(result, lev, dropna=dropna, sort=sort)  # type: ignore[assignment]

    # Otherwise, level numbers may change as each successive level is stacked
    elif all(isinstance(lev, int) for lev in level):
        # As each stack is done, the level numbers decrease, so we need
        #  to account for that when level is a sequence of ints
        result = frame
        # _get_level_number() checks level numbers are in range and converts
        # negative numbers to positive
        level = [frame.columns._get_level_number(lev) for lev in level]

        while level:
            lev = level.pop(0)
            # error: Incompatible types in assignment (expression has type
            # "Series | DataFrame", variable has type "DataFrame")
            result = stack(result, lev, dropna=dropna, sort=sort)  # type: ignore[assignment]
            # Decrement all level numbers greater than current, as these
            # have now shifted down by one
            level = [v if v <= lev else v - 1 for v in level]

    else:
        raise ValueError(
            "level should contain all level names or all level "
            "numbers, not a mixture of the two."
        )

    return result


def _stack_multi_column_index(columns: MultiIndex) -> MultiIndex | Index:
    # [FUNCTION DOC]
    # HELPER: for the multi-column stack; computes the remaining column labels after the
    # last level is removed.
    # Input: columns (hierarchical column index). With at most two levels returns the
    # first level as a plain index named by the first name. Otherwise takes the label
    # tuples of all but the last level, removes consecutive duplicates, and rebuilds a
    # hierarchical index with each level's original dtype kept (unless a level contains
    # missing placeholders, which cannot all be typed). Returns that index.
    """Creates a MultiIndex from the first N-1 levels of this MultiIndex."""
    if len(columns.levels) <= 2:
        return columns.levels[0]._rename(name=columns.names[0])

    levs = (
        [lev[c] if c >= 0 else None for c in codes]
        for lev, codes in zip(columns.levels[:-1], columns.codes[:-1], strict=True)
    )

    # Remove duplicate tuples in the MultiIndex.
    tuples = zip(*levs, strict=True)
    unique_tuples = (key for key, _ in itertools.groupby(tuples))
    new_levs = zip(*unique_tuples, strict=True)

    # The dtype of each level must be explicitly set to avoid inferring the wrong type.
    # See GH-36991.
    return MultiIndex.from_arrays(
        [
            # Not all indices can accept None values.
            Index(new_lev, dtype=lev.dtype) if None not in new_lev else new_lev
            for new_lev, lev in zip(new_levs, columns.levels[:-1], strict=True)
        ],
        names=columns.names[:-1],
    )


def _stack_multi_columns(
    frame: DataFrame, level_num: int = -1, dropna: bool = True, sort: bool = True
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: for stack; stacks one level of hierarchical columns into the row index.
    # Inputs: frame (with hierarchical columns), level_num (position of the level to
    # stack, default last), dropna (default true; drop all-missing rows from the
    # result), sort (default true; sort the columns if not lexsorted, and the stacked
    # labels).
    # Algorithm: works on a shallow copy; rolls the chosen level to the last position by
    # swapping adjacent levels (being careful when integer positions collide with level
    # names); sorts the columns if needed; computes the new columns from the other
    # levels; for each new column finds the block of source columns and builds one
    # flattened column of values (reindexing when some stacked labels are absent; for
    # extension dtypes concatenating and taking in the right order); drops new columns
    # that had no source; builds the new row index by repeating the original index codes
    # and tiling the stacked level; constructs the result, reorders columns to match the
    # expected order, and optionally drops all-missing rows.
    # Returns a DataFrame.
    def _convert_level_number(level_num: int, columns: Index):
        # [FUNCTION DOC]
        # HELPER: nested in the multi-column stack; if the integer matches a name
        # of one of the column levels, returns the name found at that position (so swaps
        # address the intended level), otherwise returns the integer unchanged.
        """
        Logic for converting the level number to something we can safely pass
        to swaplevel.

        If `level_num` matches a column name return the name from
        position `level_num`, otherwise return `level_num`.
        """
        if level_num in columns.names:
            return columns.names[level_num]

        return level_num

    this = frame.copy(deep=False)
    mi_cols = this.columns  # cast(MultiIndex, this.columns)
    assert isinstance(mi_cols, MultiIndex)  # caller is responsible

    # this makes life much simpler
    if level_num != mi_cols.nlevels - 1:
        # roll levels to put selected level at end
        roll_columns = mi_cols
        for i in range(level_num, mi_cols.nlevels - 1):
            # Need to check if the ints conflict with level names
            lev1 = _convert_level_number(i, roll_columns)
            lev2 = _convert_level_number(i + 1, roll_columns)
            roll_columns = roll_columns.swaplevel(lev1, lev2)
        this.columns = mi_cols = roll_columns

    if not mi_cols._is_lexsorted() and sort:
        # Workaround the edge case where 0 is one of the column names,
        # which interferes with trying to sort based on the first
        # level
        level_to_sort = _convert_level_number(0, mi_cols)
        this = this.sort_index(level=level_to_sort, axis=1)
        mi_cols = this.columns

    mi_cols = cast(MultiIndex, mi_cols)
    new_columns = _stack_multi_column_index(mi_cols)

    # time to ravel the values
    new_data = {}
    level_vals = mi_cols.levels[-1]
    level_codes = unique(mi_cols.codes[-1])
    if sort:
        level_codes = np.sort(level_codes)
    level_vals_nan = level_vals.insert(len(level_vals), None)

    level_vals_used = np.take(level_vals_nan, level_codes)
    levsize = len(level_codes)
    drop_cols = []
    for key in new_columns:
        try:
            loc = this.columns.get_loc(key)
        except KeyError:
            drop_cols.append(key)
            continue

        # can make more efficient?
        # we almost always return a slice
        # but if unsorted can get a boolean
        # indexer
        if not isinstance(loc, slice):
            slice_len = len(loc)
        else:
            slice_len = loc.stop - loc.start

        if slice_len != levsize:
            chunk = this.loc[:, this.columns[loc]]
            chunk.columns = level_vals_nan.take(chunk.columns.codes[-1])
            value_slice = chunk.reindex(columns=level_vals_used).values
        else:
            subset = this.iloc[:, loc]
            dtype = find_common_type(subset.dtypes.tolist())
            if isinstance(dtype, ExtensionDtype):
                # TODO(EA2D): won't need special case, can go through .values
                #  paths below (might change to ._values)
                value_slice = dtype.construct_array_type()._concat_same_type(
                    [x._values.astype(dtype, copy=False) for _, x in subset.items()]
                )
                N, K = subset.shape
                idx = np.arange(N * K).reshape(K, N).T.reshape(-1)
                value_slice = value_slice.take(idx)
            else:
                value_slice = subset.values

        if value_slice.ndim > 1:
            # i.e. not extension
            value_slice = value_slice.ravel()

        new_data[key] = value_slice

    if len(drop_cols) > 0:
        new_columns = new_columns.difference(drop_cols)

    N = len(this)

    if isinstance(this.index, MultiIndex):
        new_levels = list(this.index.levels)
        new_names = list(this.index.names)
        new_codes = [lab.repeat(levsize) for lab in this.index.codes]
    else:
        old_codes, old_levels = factorize_from_iterable(this.index)
        new_levels = [old_levels]
        new_codes = [old_codes.repeat(levsize)]
        new_names = [this.index.name]  # something better?

    new_levels.append(level_vals)
    new_codes.append(np.tile(level_codes, N))
    new_names.append(frame.columns.names[level_num])

    new_index = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    result = frame._constructor(new_data, index=new_index, columns=new_columns)

    if frame.columns.nlevels > 1:
        desired_columns = frame.columns._drop_level_numbers([level_num]).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    # more efficient way to go about this? can do the whole masking biz but
    # will only save a small amount of time...
    if dropna:
        result = result.dropna(axis=0, how="all")

    return result


def _reorder_for_extension_array_stack(
    arr: ExtensionArray, n_rows: int, n_columns: int
) -> ExtensionArray:
    # [FUNCTION DOC]
    # HELPER: for stack; reorders the values produced by concatenating the columns of a
    # homogeneous extension-array frame column by column into row-major order.
    # Inputs: arr (concatenated values), n_rows and n_columns (shape of the original
    # frame). Builds a permutation that interleaves the columns row by row and returns a
    # new array taken in that order.
    """
    Re-orders the values when stacking multiple extension-arrays.

    The indirect stacking method used for EAs requires a followup
    take to get the order correct.

    Parameters
    ----------
    arr : ExtensionArray
    n_rows, n_columns : int
        The number of rows and columns in the original DataFrame.

    Returns
    -------
    taken : ExtensionArray
        The original `arr` with elements re-ordered appropriately

    Examples
    --------
    >>> arr = np.array(["a", "b", "c", "d", "e", "f"])
    >>> _reorder_for_extension_array_stack(arr, 2, 3)
    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')

    >>> _reorder_for_extension_array_stack(arr, 3, 2)
    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')
    """
    # final take to get the order correct.
    # idx is an indexer like
    # [c0r0, c1r0, c2r0, ...,
    #  c0r1, c1r1, c2r1, ...]
    idx = np.arange(n_rows * n_columns).reshape(n_columns, n_rows).T.reshape(-1)
    return arr.take(idx)


def stack_v3(frame: DataFrame, level: list[int]) -> Series | DataFrame:
    # [FUNCTION DOC]
    # Newer stack implementation: moves the given column levels into the row index.
    # Inputs: frame; level (list of integer column-level positions to stack).
    # Raises a value error if the column labels contain duplicates. Returns the frame
    # unchanged when level is empty.
    # Algorithm: computes the stacked columns (the dropped-out other levels); for flat
    # columns uses a fast path reshaping the values column-major into a Series (keeping a
    # common dtype for homogeneous frames); otherwise uses the stack reshape helper.
    # Then builds the row index by combining the original index (tiled by the stack
    # ratio) with the unique stacked column labels (in the requested level order), and
    # reorders the result with a computed permutation so rows are grouped by original row.
    # If all column levels were stacked, a single-column result is reduced to a Series
    # (an empty one when there are no columns); a Series result has its name cleared.
    # Returns a Series or DataFrame.
    if frame.columns.nunique() != len(frame.columns):
        raise ValueError("Columns with duplicate values are not supported in stack")
    if not len(level):
        return frame
    set_levels = set(level)
    stack_cols = frame.columns._drop_level_numbers(
        [k for k in range(frame.columns.nlevels - 1, -1, -1) if k not in set_levels]
    )

    result: Series | DataFrame
    if not isinstance(frame.columns, MultiIndex):
        # GH#58817 Fast path when we're stacking the columns of a non-MultiIndex.
        # When columns are homogeneous EAs, we pass through object
        # dtype but this is still slightly faster than the normal path.
        if len(frame.columns) > 0 and frame._is_homogeneous_type:
            dtype = frame._mgr.blocks[0].dtype
        else:
            dtype = None
        result = frame._constructor_sliced(
            frame._values.reshape(-1, order="F"), dtype=dtype
        )
    else:
        result = stack_reshape(frame, level, set_levels, stack_cols)

    # Construct the correct MultiIndex by combining the frame's index and
    # stacked columns.
    ratio = 0 if frame.empty else len(result) // len(frame)

    index_levels: list | FrozenList
    if isinstance(frame.index, MultiIndex):
        index_levels = frame.index.levels
        index_codes = list(np.tile(frame.index.codes, (1, ratio)))
    else:
        codes, uniques = factorize(frame.index, use_na_sentinel=False)
        index_levels = [uniques]
        index_codes = list(np.tile(codes, (1, ratio)))

    if len(level) > 1:
        # Arrange columns in the order we want to take them, e.g. level=[2, 0, 1]
        sorter = np.argsort(level)
        assert isinstance(stack_cols, MultiIndex)
        ordered_stack_cols = stack_cols._reorder_ilevels(sorter)
    else:
        ordered_stack_cols = stack_cols
    ordered_stack_cols_unique = ordered_stack_cols.unique()
    if isinstance(ordered_stack_cols, MultiIndex):
        column_levels = ordered_stack_cols.levels
        column_codes = ordered_stack_cols.drop_duplicates().codes
    else:
        column_levels = [ordered_stack_cols_unique]
        column_codes = [factorize(ordered_stack_cols_unique, use_na_sentinel=False)[0]]

    # error: Incompatible types in assignment (expression has type "list[ndarray[Any,
    # dtype[Any]]]", variable has type "FrozenList")
    column_codes = [np.repeat(codes, len(frame)) for codes in column_codes]  # type: ignore[assignment]
    result.index = MultiIndex(
        levels=index_levels + column_levels,
        codes=index_codes + column_codes,
        names=frame.index.names + list(ordered_stack_cols.names),
        verify_integrity=False,
    )

    # sort result, but faster than calling sort_index since we know the order we need
    len_df = len(frame)
    n_uniques = len(ordered_stack_cols_unique)
    indexer = np.arange(n_uniques)
    idxs = np.tile(len_df * indexer, len_df) + np.repeat(np.arange(len_df), n_uniques)
    result = result.take(idxs)

    # Reshape/rename if needed and dropna
    if result.ndim == 2 and frame.columns.nlevels == len(level):
        if len(result.columns) == 0:
            result = Series(index=result.index)
        else:
            result = result.iloc[:, 0]
    if result.ndim == 1:
        result.name = None

    return result


def stack_reshape(
    frame: DataFrame, level: list[int], set_levels: set[int], stack_cols: Index
) -> Series | DataFrame:
    # [FUNCTION DOC]
    # HELPER: for stack_v3; does the data movement for stacking levels of hierarchical
    # columns. Inputs: frame (hierarchical columns, asserted), level (positions to
    # stack), set_levels (same as a set), stack_cols (labels of the stacked levels, one per
    # column). For each distinct stacked label, selects the matching columns (the whole
    # frame when it has one column), drops the stacked levels from their column index (or,
    # when all levels are stacked, resets the columns to a default range or names a Series
    # zero), and collects the pieces; the pieces are concatenated with a fresh index.
    # For empty input builds an empty frame with the original dtype. If some levels remain,
    # reorders columns to the unique order of the remaining levels. Returns the data frame
    # (the caller sets the final index and sorts).
    # KNOWN ISSUE: the docstring's Returns text has a typo ("data of behind") but is
    # otherwise consistent with the code.
    """Reshape the data of a frame for stack.

    This function takes care of most of the work that stack needs to do. Caller
    will sort the result once the appropriate index is set.

    Parameters
    ----------
    frame: DataFrame
        DataFrame that is to be stacked.
    level: list of ints.
        Levels of the columns to stack.
    set_levels: set of ints.
        Same as level, but as a set.
    stack_cols: Index.
        Columns of the result when the DataFrame is stacked.

    Returns
    -------
    The data of behind the stacked DataFrame.
    """
    # non-MultIndex takes a fast path.
    assert isinstance(frame.columns, MultiIndex)
    # If we need to drop `level` from columns, it needs to be in descending order
    drop_levnums = sorted(level, reverse=True)

    # Grab data for each unique index to be stacked
    buf = []
    for idx in stack_cols.unique():
        if len(frame.columns) == 1:
            data = frame.copy(deep=False)
        else:
            # Take the data from frame corresponding to this idx value
            if len(level) == 1:
                idx = (idx,)
            gen = iter(idx)
            column_indexer = tuple(
                next(gen) if k in set_levels else slice(None)
                for k in range(frame.columns.nlevels)
            )
            data = frame.loc[:, column_indexer]

        if len(level) < frame.columns.nlevels:
            data.columns = data.columns._drop_level_numbers(drop_levnums)
        elif stack_cols.nlevels == 1:
            if data.ndim == 1:
                data.name = 0
            else:
                data.columns = default_index(len(data.columns))
        buf.append(data)

    if len(buf) > 0 and not frame.empty:
        result = concat(buf, ignore_index=True)
    else:
        # input is empty
        if len(level) < frame.columns.nlevels:
            # concat column order may be different from dropping the levels
            new_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        else:
            new_columns = [0]
        result = DataFrame(columns=new_columns, dtype=frame._values.dtype)

    if len(level) < frame.columns.nlevels:
        # concat column order may be different from dropping the levels
        desired_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    return result
