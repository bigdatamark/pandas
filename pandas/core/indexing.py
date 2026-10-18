# [MODULE DOC]
# Implementation of the label- and position-based indexers (loc, iloc, at, iat)
# for frames and series, including the machinery for getting and setting with
# all supported key forms.
# Public contents: the slicing helper object for multi-level indexes (a single
# instance exposed as the public name for building multi-index slices), the
# mixin that adds the four indexer properties to frames and series, the abstract
# base for the loc/iloc indexers, the concrete loc, iloc, scalar-label and
# scalar-position indexers, the shared scalar-indexer base, and the helper
# functions for converting keys (non-numeric/positional checks, conversion of
# 2D indexers, slice and mask handling, and a check for out-of-bounds
# positional indexers).
# Private module constants: a full slice, and the message used when more than one
# ellipsis is given. Collaborates with the compiled indexer base class,
# the block manager (setitem/take), the index classes (label lookup), and
# the copy-on-write chained assignment warnings. No import-time side effects
# other than creating the public slicer object and setting its module name.
from __future__ import annotations

from contextlib import suppress
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import CHAINED_WARNING_DISABLED
from pandas.compat._constants import REF_COUNT_IDX
from pandas.errors import (
    AbstractMethodError,
    ChainedAssignmentError,
    IndexingError,
    InvalidIndexError,
    LossySetitemError,
)
from pandas.errors.cow import _chained_assignment_msg
from pandas.util._decorators import (
    doc,
)

from pandas.core.dtypes.cast import (
    can_hold_element,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_sequence,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    construct_1d_array_from_inferred_fill_value,
    infer_fill_value,
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    is_list_like_indexer,
    is_scalar_indexer,
    length_of_indexer,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        Axis,
        AxisInt,
        T,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )

# "null slice"
_NS = slice(None, None)
_one_ellipsis_message = "indexer may only contain one '...' entry"


# the public IndexSlicerMaker
class _IndexSlice:
    # [CLASS DOC]
    # Helper that makes multi-index slicing easier to write: indexing an instance
    # returns the key unchanged (so slice and tuple syntax can be used outside
    # of brackets of a frame). Stateless. A single public instance is created below.
    """
    Create an object to more easily perform multi-index slicing.

    See Also
    --------
    MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.

    Notes
    -----
    See :ref:`Defined Levels <advanced.shown_levels>`
    for further info on slicing a MultiIndex.

    Examples
    --------
    >>> midx = pd.MultiIndex.from_product([["A0", "A1"], ["B0", "B1", "B2", "B3"]])
    >>> columns = ["foo", "bar"]
    >>> dfmi = pd.DataFrame(
    ...     np.arange(16).reshape((len(midx), len(columns))),
    ...     index=midx,
    ...     columns=columns,
    ... )

    Using the default slice command:

    >>> dfmi.loc[(slice(None), slice("B0", "B1")), :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11

    Using the IndexSlice class for a more intuitive command:

    >>> idx = pd.IndexSlice
    >>> dfmi.loc[idx[:, "B0":"B1"], :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11
    """

    def __getitem__(self, arg):
        # [FUNCTION DOC]
        # Return the argument unchanged. Parameter arg: whatever was passed in brackets
        # (slices, tuples, mixtures).
        return arg


IndexSlice = _IndexSlice()
IndexSlice.__module__ = "pandas"


class IndexingMixin:
    # [CLASS DOC]
    # Mixin that gives frames and series the four indexer accessors: label-based
    # (loc), position-based (iloc), scalar label-based (at) and scalar position-
    # based (iat). Stateless; each property creates a new indexer object bound to
    # the owning object and a string naming the indexer.
    """
    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.
    """

    @property
    def iloc(self) -> _iLocIndexer:
        # [FUNCTION DOC]
        # Property returning a new position-based indexer for the owner, named iloc.
        # Allowed keys: integer, list or array of integers, slice, boolean array,
        # callable (returning one of those) and tuples of those for rows and columns.
        # Out-of-bounds integers raise an index error, out-of-bounds slices do not.
        """
        Purely integer-location based indexing for selection by position.

        .. versionchanged:: 3.0

           Callables which return a tuple are deprecated as input.

        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean
        array.

        Allowed inputs are:

        - An integer, e.g. ``5``.
        - A list or array of integers, e.g. ``[4, 3, 0]``.
        - A slice object with ints, e.g. ``1:7``.
        - A boolean array.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above).
          This is useful in method chains, when you don't have a reference to the
          calling object, but would like to base your selection on
          some value.
        - A tuple of row and column indexes. The tuple elements consist of one of the
          above inputs, e.g. ``(0, 1)``.

        ``.iloc`` will raise ``IndexError`` if a requested indexer is
        out-of-bounds, except *slice* indexers which allow out-of-bounds
        indexing (this conforms with python/numpy *slice* semantics).

        See more at :ref:`Selection by Position <indexing.integer>`.

        See Also
        --------
        DataFrame.iat : Fast integer location scalar accessor.
        DataFrame.loc : Purely label-location based indexer for selection by label.
        Series.iloc : Purely integer-location based indexing for
                       selection by position.

        Examples
        --------
        >>> mydict = [
        ...     {"a": 1, "b": 2, "c": 3, "d": 4},
        ...     {"a": 100, "b": 200, "c": 300, "d": 400},
        ...     {"a": 1000, "b": 2000, "c": 3000, "d": 4000},
        ... ]
        >>> df = pd.DataFrame(mydict)
        >>> df
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        **Indexing just the rows**

        With a scalar integer.

        >>> type(df.iloc[0])
        <class 'pandas.Series'>
        >>> df.iloc[0]
        a    1
        b    2
        c    3
        d    4
        Name: 0, dtype: int64

        With a list of integers.

        >>> df.iloc[[0]]
           a  b  c  d
        0  1  2  3  4
        >>> type(df.iloc[[0]])
        <class 'pandas.DataFrame'>

        >>> df.iloc[[0, 1]]
             a    b    c    d
        0    1    2    3    4
        1  100  200  300  400

        With a `slice` object.

        >>> df.iloc[:3]
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        With a boolean mask the same length as the index.

        >>> df.iloc[[True, False, True]]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        With a callable, useful in method chains. The `x` passed
        to the ``lambda`` is the DataFrame being sliced. This selects
        the rows whose index label even.

        >>> df.iloc[lambda x: x.index % 2 == 0]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        **Indexing both axes**

        You can mix the indexer types for the index and columns. Use ``:`` to
        select the entire axis.

        With scalar integers.

        >>> df.iloc[0, 1]
        np.int64(2)

        With lists of integers.

        >>> df.iloc[[0, 2], [1, 3]]
              b     d
        0     2     4
        2  2000  4000

        With `slice` objects.

        >>> df.iloc[1:3, 0:3]
              a     b     c
        1   100   200   300
        2  1000  2000  3000

        With a boolean array whose length matches the columns.

        >>> df.iloc[:, [True, False, True, False]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000

        With a callable function that expects the Series or DataFrame.

        >>> df.iloc[:, lambda df: [0, 2]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000
        """
        return _iLocIndexer("iloc", self)

    @property
    def loc(self) -> _LocIndexer:
        # [FUNCTION DOC]
        # Property returning a new label-based indexer for the owner, named loc.
        # Allowed keys: label, list or array of labels, slice of labels (stop
        # included), boolean array, alignable boolean series or index, callable, and
        # tuples for rows and columns (including multi-index tuples). Missing labels
        # raise a key error when getting; keys that cannot be aligned raise an indexing
        # error. Setting with a series or frame aligns on labels.
        """
        Access a group of rows and columns by label(s) or a boolean array.

        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.

        Allowed inputs are:

        - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
          interpreted as a *label* of the index, and **never** as an
          integer position along the index).
        - A list or array of labels, e.g. ``['a', 'b', 'c']``.
        - A slice object with labels, e.g. ``'a':'f'``.

          .. warning:: Note that contrary to usual python slices, **both** the
              start and the stop are included

        - A boolean array of the same length as the axis being sliced,
          e.g. ``[True, False, True]``.
        - An alignable boolean Series. The index of the key will be aligned before
          masking.
        - An alignable Index. The Index of the returned selection will be the input.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above)

        See more at :ref:`Selection by Label <indexing.label>`.

        Raises
        ------
        KeyError
            If any items are not found.
        IndexingError
            If an indexed key is passed and its index is unalignable to the frame index.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.iloc : Access group of rows and columns by integer position(s).
        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
                       Series/DataFrame.
        Series.loc : Access group of values using labels.

        Examples
        --------
        **Getting values**

        >>> df = pd.DataFrame(
        ...     [[1, 2], [4, 5], [7, 8]],
        ...     index=["cobra", "viper", "sidewinder"],
        ...     columns=["max_speed", "shield"],
        ... )
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8

        Single label. Note this returns the row as a Series.

        >>> df.loc["viper"]
        max_speed    4
        shield       5
        Name: viper, dtype: int64

        List of labels. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[["viper", "sidewinder"]]
                    max_speed  shield
        viper               4       5
        sidewinder          7       8

        Single label for row and column

        >>> df.loc["cobra", "shield"]
        np.int64(2)

        Slice with labels for row and single label for column. As mentioned
        above, note that both the start and stop of the slice are included.

        >>> df.loc["cobra":"viper", "max_speed"]
        cobra    1
        viper    4
        Name: max_speed, dtype: int64

        Boolean list with the same length as the row axis

        >>> df.loc[[False, False, True]]
                    max_speed  shield
        sidewinder          7       8

        Alignable boolean Series:

        >>> df.loc[
        ...     pd.Series([False, True, False], index=["viper", "sidewinder", "cobra"])
        ... ]
                             max_speed  shield
        sidewinder          7       8

        Index (same behavior as ``df.reindex``)

        >>> df.loc[pd.Index(["cobra", "viper"], name="foo")]
               max_speed  shield
        foo
        cobra          1       2
        viper          4       5

        Conditional that returns a boolean Series

        >>> df.loc[df["shield"] > 6]
                    max_speed  shield
        sidewinder          7       8

        Conditional that returns a boolean Series with column labels specified

        >>> df.loc[df["shield"] > 6, ["max_speed"]]
                    max_speed
        sidewinder          7

        Multiple conditional using ``&`` that returns a boolean Series

        >>> df.loc[(df["max_speed"] > 1) & (df["shield"] < 8)]
                    max_speed  shield
        viper          4       5

        Multiple conditional using ``|`` that returns a boolean Series

        >>> df.loc[(df["max_speed"] > 4) | (df["shield"] < 5)]
                    max_speed  shield
        cobra               1       2
        sidewinder          7       8

        Please ensure that each condition is wrapped in parentheses ``()``.
        See the :ref:`user guide<indexing.boolean>`
        for more details and explanations of Boolean indexing.

        .. note::
            If you find yourself using 3 or more conditionals in ``.loc[]``,
            consider using :ref:`advanced indexing<advanced.advanced_hierarchical>`.

            See below for using ``.loc[]`` on MultiIndex DataFrames.

        Callable that returns a boolean Series

        >>> df.loc[lambda df: df["shield"] == 8]
                    max_speed  shield
        sidewinder          7       8

        **Setting values**

        Set value for all items matching the list of labels

        >>> df.loc[["viper", "sidewinder"], ["shield"]] = 50
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4      50
        sidewinder          7      50

        Set value for an entire row

        >>> df.loc["cobra"] = 10
        >>> df
                    max_speed  shield
        cobra              10      10
        viper               4      50
        sidewinder          7      50

        Set value for an entire column

        >>> df.loc[:, "max_speed"] = 30
        >>> df
                    max_speed  shield
        cobra              30      10
        viper              30      50
        sidewinder         30      50

        Set value for rows matching callable condition

        >>> df.loc[df["shield"] > 35] = 0
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       0
        sidewinder          0       0

        Add value matching location

        >>> df.loc["viper", "shield"] += 5
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       5
        sidewinder          0       0

        Setting using a ``Series`` or a ``DataFrame`` sets the values matching the
        index labels, not the index positions.

        >>> shuffled_df = df.loc[["viper", "cobra", "sidewinder"]]
        >>> df.loc[:] += shuffled_df
        >>> df
                    max_speed  shield
        cobra              60      20
        viper               0      10
        sidewinder          0       0

        **Getting values on a DataFrame with an index that has integer labels**

        Another example using integers for the index

        >>> df = pd.DataFrame(
        ...     [[1, 2], [4, 5], [7, 8]],
        ...     index=[7, 8, 9],
        ...     columns=["max_speed", "shield"],
        ... )
        >>> df
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        Slice with integer labels for rows. As mentioned above, note that both
        the start and stop of the slice are included.

        >>> df.loc[7:9]
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        **Getting values with a MultiIndex**

        A number of examples using a DataFrame with a MultiIndex

        >>> tuples = [
        ...     ("cobra", "mark i"),
        ...     ("cobra", "mark ii"),
        ...     ("sidewinder", "mark i"),
        ...     ("sidewinder", "mark ii"),
        ...     ("viper", "mark ii"),
        ...     ("viper", "mark iii"),
        ... ]
        >>> index = pd.MultiIndex.from_tuples(tuples)
        >>> values = [[12, 2], [0, 4], [10, 20], [1, 4], [7, 1], [16, 36]]
        >>> df = pd.DataFrame(values, columns=["max_speed", "shield"], index=index)
        >>> df
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Single label. Note this returns a DataFrame with a single index.

        >>> df.loc["cobra"]
                 max_speed  shield
        mark i          12       2
        mark ii          0       4

        Single index tuple. Note this returns a Series.

        >>> df.loc[("cobra", "mark ii")]
        max_speed    0
        shield       4
        Name: (cobra, mark ii), dtype: int64

        Single label for row and column. Similar to passing in a tuple, this
        returns a Series.

        >>> df.loc["cobra", "mark i"]
        max_speed    12
        shield        2
        Name: (cobra, mark i), dtype: int64

        Single tuple. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[[("cobra", "mark ii")]]
                       max_speed  shield
        cobra mark ii          0       4

        Single tuple for the index with a single label for the column

        >>> df.loc[("cobra", "mark i"), "shield"]
        np.int64(2)

        Slice from index tuple to single label

        >>> df.loc[("cobra", "mark i") : "viper"]
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Slice from index tuple to index tuple

        >>> df.loc[("cobra", "mark i") : ("viper", "mark ii")]
                            max_speed  shield
        cobra      mark i          12       2
                   mark ii          0       4
        sidewinder mark i          10      20
                   mark ii          1       4
        viper      mark ii          7       1

        Please see the :ref:`user guide<advanced.advanced_hierarchical>`
        for more details and explanations of advanced indexing.

        **Assignment with Series**

        When assigning a Series to .loc[row_indexer, col_indexer], pandas aligns
        the Series by index labels, not by order or position.

        Series assignment with .loc and index alignment:

        >>> df = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
        >>> s = pd.Series([10, 20], index=[1, 0])  # Note reversed order
        >>> df.loc[:, "B"] = s  # Aligns by index, not order
        >>> df
           A   B
        0  1  20.0
        1  2  10.0
        2  3 NaN
        """
        return _LocIndexer("loc", self)

    @property
    def at(self) -> _AtIndexer:
        # [FUNCTION DOC]
        # Property returning a new scalar label-based indexer for the owner, named at.
        # Used to get or set a single value by row and column label. Errors: a key error
        # for missing labels when getting; a value error if the pair is not a tuple or
        # is not scalar for frames, or if the label is list-like for series.
        """
        Access a single value for a row/column label pair.

        Similar to ``loc``, in that both provide label-based lookups. Use
        ``at`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        KeyError
            If getting a value and 'label' does not exist in a DataFrame or Series.

        ValueError
            If row/column label pair is not a tuple or if any label
            from the pair is not a scalar for DataFrame.
            If label is list-like (*excluding* NamedTuple) for Series.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column pair by label.
        DataFrame.iat : Access a single value for a row/column pair by integer
            position.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer
            position(s).
        Series.at : Access a single value by label.
        Series.iat : Access a single value by integer position.
        Series.loc : Access a group of rows by label(s).
        Series.iloc : Access a group of rows by integer position(s).

        Notes
        -----
        See :ref:`Fast scalar value getting and setting <indexing.basics.get_value>`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [[0, 2, 3], [0, 4, 1], [10, 20, 30]],
        ...     index=[4, 5, 6],
        ...     columns=["A", "B", "C"],
        ... )
        >>> df
            A   B   C
        4   0   2   3
        5   0   4   1
        6  10  20  30

        Get value at specified row/column pair

        >>> df.at[4, "B"]
        np.int64(2)

        Set value at specified row/column pair

        >>> df.at[4, "B"] = 10
        >>> df.at[4, "B"]
        np.int64(10)

        Get value within a Series

        >>> df.loc[5].at["B"]
        np.int64(4)
        """
        return _AtIndexer("at", self)

    @property
    def iat(self) -> _iAtIndexer:
        # [FUNCTION DOC]
        # Property returning a new scalar position-based indexer for the owner, named
        # iat. Used to get or set a single value by integer positions. Raises an
        # index error for out-of-bounds positions.
        """
        Access a single value for a row/column pair by integer position.

        Similar to ``iloc``, in that both provide integer-based lookups. Use
        ``iat`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        IndexError
            When integer position is out of bounds.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer position(s).

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [[0, 2, 3], [0, 4, 1], [10, 20, 30]], columns=["A", "B", "C"]
        ... )
        >>> df
            A   B   C
        0   0   2   3
        1   0   4   1
        2  10  20  30

        Get value at specified row/column pair

        >>> df.iat[1, 2]
        np.int64(1)

        Set value at specified row/column pair

        >>> df.iat[1, 2] = 10
        >>> df.iat[1, 2]
        np.int64(10)

        Get value within a series

        >>> df.loc[0].iat[1]
        np.int64(2)
        """
        return _iAtIndexer("iat", self)


class _LocationIndexer(NDFrameIndexerBase):
    # [CLASS DOC]
    # Abstract base class of the loc and iloc indexers; subclass of the compiled
    # indexer base (which stores the owner as the obj attribute, the indexer name
    # as the name attribute and the ndim). Class-level state: a description of
    # valid key types (set by subclasses), an optional axis restriction (default
    # none, set via calling the indexer with an axis) and a flag saying whether
    # integer keys are positions (set by subclasses). Provides get/set entry points,
    # tuple handling (ellipsis expansion, key length validation, lower-dimension
    # and nested-tuple retrieval), set-indexer conversion, and abstract hooks for
    # subclasses (key validation, axis lookup, conversion to positional indexers).
    _valid_types: str
    axis: AxisInt | None = None

    # sub-classes need to set _takeable
    _takeable: bool

    @final
    def __call__(self, axis: Axis | None = None) -> Self:
        # [FUNCTION DOC]
        # Final. Allow calling the indexer to restrict it to one axis. Parameter axis:
        # optional axis name or number. Returns a new indexer of the same class and
        # name over the same object, with its axis restriction set to the axis number
        # (or left None).
        # we need to return a copy of ourselves
        new_self = type(self)(self.name, self.obj)

        if axis is not None:
            axis_int_none = self.obj._get_axis_number(axis)
        else:
            axis_int_none = axis
        new_self.axis = axis_int_none
        return new_self

    def _get_setitem_indexer(self, key):
        # [FUNCTION DOC]
        # HELPER: convert a possibly label-based key to a positional indexer for setting.
        # Parameter key. For label indexers first makes sure that listed column labels
        # exist (adding missing ones), validates tuple elements are not dicts or sets,
        # and expresses a one-axis restriction as a full tuple. Multi-index frames
        # with a hashable non-slice key (label indexers only) try a direct location
        # lookup first, ignoring key and invalid-index errors. Tuple keys are converted
        # per axis, ignoring a "too many indexers" error; ranges become lists;
        # anything else is converted on the first axis. Returns the positional indexer.
        """
        Convert a potentially-label-based key into a positional indexer.
        """
        if self.name == "loc":
            # always holds here bc iloc overrides _get_setitem_indexer
            self._ensure_listlike_indexer(key, axis=self.axis)

        if isinstance(key, tuple):
            for x in key:
                check_dict_or_set_indexers(x)

        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)

        ax = self.obj._get_axis(0)

        if (
            isinstance(ax, MultiIndex)
            and self.name != "iloc"
            and is_hashable(key, allow_slice=False)
        ):
            with suppress(KeyError, InvalidIndexError):
                # TypeError e.g. passed a bool
                return ax.get_loc(key)

        if isinstance(key, tuple):
            with suppress(IndexingError):
                # suppress "Too many indexers"
                return self._convert_tuple(key)

        if isinstance(key, range):
            # GH#45479 test_loc_setitem_range_key
            key = list(key)

        return self._convert_to_indexer(key, axis=0)

    @final
    def _maybe_mask_setitem_value(self, indexer, value):
        # [FUNCTION DOC]
        # Final. HELPER: for setting with a series or frame value and a (boolean mask,
        # column) tuple indexer where the value has the same length as the mask, treat
        # the value as the one aligned to the selected rows, similar to series
        # setitem. Parameters: indexer; value. For a scalar column indexer and series
        # value, the series is aligned to the indexer (or truncated if the mask selects
        # nothing) and the mask becomes positions; for a one-element integer array
        # column indexer, a series is aligned likewise and a one-column frame value is
        # aligned or truncated. A bare boolean indexer becomes positions. Returns a
        # pair of (indexer, value).
        # KNOWN ISSUE: the existing docstring talks about setting through position-based access with a mask, but also says it is only for label-based access; it is only used for label-based access.
        """
        If we have obj.iloc[mask] = series_or_frame and series_or_frame has the
        same length as obj, we treat this as obj.iloc[mask] = series_or_frame[mask],
        similar to Series.__setitem__.

        Note this is only for loc, not iloc.
        """

        if (
            isinstance(indexer, tuple)
            and len(indexer) == 2
            and isinstance(value, (ABCSeries, ABCDataFrame))
        ):
            pi, icols = indexer
            ndim = value.ndim
            if com.is_bool_indexer(pi) and len(value) == len(pi):
                newkey = pi.nonzero()[0]

                if is_scalar_indexer(icols, self.ndim - 1) and ndim == 1:
                    # e.g. test_loc_setitem_boolean_mask_allfalse
                    if len(newkey) == 0:
                        value = value.iloc[:0]
                    else:
                        # test_loc_setitem_ndframe_values_alignment
                        value = self.obj.iloc._align_series(indexer, value)
                    indexer = (newkey, icols)

                elif (
                    isinstance(icols, np.ndarray)
                    and icols.dtype.kind == "i"
                    and len(icols) == 1
                ):
                    if ndim == 1:
                        # We implicitly broadcast, though numpy does not, see
                        # github.com/pandas-dev/pandas/pull/45501#discussion_r789071825
                        # test_loc_setitem_ndframe_values_alignment
                        value = self.obj.iloc._align_series(indexer, value)
                        indexer = (newkey, icols)

                    elif ndim == 2 and value.shape[1] == 1:
                        if len(newkey) == 0:
                            value = value.iloc[:0]
                        else:
                            # test_loc_setitem_ndframe_values_alignment
                            value = self.obj.iloc._align_frame(indexer, value)
                        indexer = (newkey, icols)
        elif com.is_bool_indexer(indexer):
            indexer = indexer.nonzero()[0]

        return indexer, value

    @final
    def _ensure_listlike_indexer(self, key, axis=None, value=None) -> None:
        # [FUNCTION DOC]
        # Final. HELPER: make sure that a list-like of column labels exists in the frame,
        # adding missing columns. Parameters: key (list-like or tuple of keys; for a
        # tuple longer than one the part for the axis, default the column axis, is used);
        # axis (optional); value (accepted but unused).
        # Only applies to 2D objects, a column axis, non-multi columns and non-boolean
        # list-like keys of hashable labels. If any labels are missing the manager is
        # reindexed to the union of columns, with the new columns as placeholders of void
        # dtype so that later setting through the single-column path decides the dtype
        # instead of creating floating columns; otherwise the manager is reindexed
        # without placeholders. Mutates the owner's manager in place. Returns nothing.
        # KNOWN ISSUE: the value parameter is never used in the body, and the existing docstring does not describe it.
        """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.

        Parameters
        ----------
        key : list-like of column labels
            Target labels.
        axis : key axis if known
        """
        column_axis = 1

        # column only exists in 2-dimensional DataFrame
        if self.ndim != 2:
            return

        if isinstance(key, tuple) and len(key) > 1:
            # key may be a tuple if we are .loc
            # if length of key is > 1 set key to column part
            # unless axis is already specified, then go with that
            if axis is None:
                axis = column_axis
            key = key[axis]

        if (
            axis == column_axis
            and not isinstance(self.obj.columns, MultiIndex)
            and is_list_like_indexer(key)
            and not com.is_bool_indexer(key)
            and all(is_hashable(k) for k in key)
        ):
            # GH#38148
            keys = self.obj.columns.union(key, sort=False)
            diff = Index(key, copy=False).difference(self.obj.columns, sort=False)

            if len(diff):
                # e.g. if we are doing df.loc[:, ["A", "B"]] = 7 and "B"
                #  is a new column, add the new columns with dtype=np.void
                #  so that later when we go through setitem_single_column
                #  we will use isetitem. Without this, the reindex_axis
                #  below would create float64 columns in this example, which
                #  would successfully hold 7, so we would end up with the wrong
                #  dtype.
                indexer = np.arange(len(keys), dtype=np.intp)
                indexer[len(self.obj.columns) :] = -1
                new_mgr = self.obj._mgr.reindex_indexer(
                    keys, indexer=indexer, axis=0, only_slice=True, use_na_proxy=True
                )
                self.obj._mgr = new_mgr
                return

            self.obj._mgr = self.obj._mgr.reindex_axis(keys, axis=0, only_slice=True)

    @final
    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Final. Set items through the indexer. Parameters: key (label, positional key,
        # callable, or tuple thereof); value.
        # If the chained assignment check is not disabled and the owner has no more
        # references than a plain local would, emits a chained assignment warning
        # (stack level two). Validates the key is not a dict or set; converts iterators
        # and callables (applying them to the owner; a callable returning a tuple from the
        # position indexer raises a value error); converts the key to a positional
        # indexer, validates it, and dispatches the set to the position indexer.
        if not CHAINED_WARNING_DISABLED:
            if sys.getrefcount(self.obj) <= REF_COUNT_IDX:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )

        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            key = (list(x) if is_iterator(x) else x for x in key)
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            maybe_callable = com.apply_if_callable(key, self.obj)
            key = self._raise_callable_usage(key, maybe_callable)
        indexer = self._get_setitem_indexer(key)
        self._has_valid_setitem_indexer(key)

        iloc: _iLocIndexer = (
            cast("_iLocIndexer", self) if self.name == "iloc" else self.obj.iloc
        )
        iloc._setitem_with_indexer(indexer, value, self.name)

    def _validate_key(self, key, axis: AxisInt) -> None:
        # [FUNCTION DOC]
        # Abstract hook: validate a key for this indexer along an axis. Parameters: key;
        # axis. Subclasses raise type, index or key errors as appropriate. The base
        # raises the abstract method error.
        """
        Ensure that key is valid for current indexer.

        Parameters
        ----------
        key : scalar, slice or list-like
            Key requested.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        TypeError
            If the key (or some element of it) has wrong type.
        IndexError
            If the key (or some element of it) is out of bounds.
        KeyError
            If the key was not found.
        """
        raise AbstractMethodError(self)

    @final
    def _expand_ellipsis(self, tup: tuple) -> tuple:
        # [FUNCTION DOC]
        # Final. HELPER: replace an ellipsis in a tuple key with null slices. Parameter
        # tup. Raises an indexing error if more than one ellipsis exists. If the tuple
        # length equals the object dimension the ellipsis becomes one null slice;
        # other cases are returned unchanged. Returns a tuple.
        """
        If a tuple key includes an Ellipsis, replace it with an appropriate
        number of null slices.
        """
        if any(x is Ellipsis for x in tup):
            if tup.count(Ellipsis) > 1:
                raise IndexingError(_one_ellipsis_message)

            if len(tup) == self.ndim:
                # It is unambiguous what axis this Ellipsis is indexing,
                #  treat as a single null slice.
                i = tup.index(Ellipsis)
                # FIXME: this assumes only one Ellipsis
                new_key = (*tup[:i], _NS, *tup[i + 1 :])
                return new_key

            # TODO: other cases?  only one test gets here, and that is covered
            #  by _validate_key_length
        return tup

    @final
    def _validate_tuple_indexer(self, key: tuple) -> tuple:
        # [FUNCTION DOC]
        # Final. HELPER: validate a tuple key. Parameter key. Checks the key length,
        # expands the ellipsis and validates each element along its axis; a value error
        # from validation is re-raised with a message listing the valid types. Returns
        # the processed tuple.
        """
        Check the key for valid keys across my indexer.
        """
        key = self._validate_key_length(key)
        key = self._expand_ellipsis(key)
        for i, k in enumerate(key):
            try:
                self._validate_key(k, i)
            except ValueError as err:
                raise ValueError(
                    f"Location based indexing can only have [{self._valid_types}] types"
                ) from err
        return key

    @final
    def _is_nested_tuple_indexer(self, tup: tuple) -> bool:
        # [FUNCTION DOC]
        # Final. HELPER: true if any axis is a multi-index and the tuple is a nested
        # tuple for any axis (according to the helper function); false otherwise.
        """
        Returns
        -------
        bool
        """
        if any(isinstance(ax, MultiIndex) for ax in self.obj.axes):
            return any(is_nested_tuple(tup, ax) for ax in self.obj.axes)
        return False

    @final
    def _convert_tuple(self, key: tuple) -> tuple:
        # [FUNCTION DOC]
        # Final. HELPER: convert each element of a tuple key to a positional indexer
        # along its own axis, after validating the key length. Returns a tuple. Assumes
        # the axis restriction was already applied to the key.
        # Note: we assume _tupleize_axis_indexer has been called, if necessary.
        self._validate_key_length(key)
        keyidx = [self._convert_to_indexer(k, axis=i) for i, k in enumerate(key)]
        return tuple(keyidx)

    @final
    def _validate_key_length(self, key: tuple) -> tuple:
        # [FUNCTION DOC]
        # Final. HELPER: validate that a tuple key is not longer than the object
        # dimension. If it is, a leading ellipsis is dropped (raising an indexing error
        # if another ellipsis is present) and the check repeated; otherwise an indexing
        # error saying there are too many indexers is raised. Returns the key.
        if len(key) > self.ndim:
            if key[0] is Ellipsis:
                # e.g. Series.iloc[..., 3] reduces to just Series.iloc[3]
                key = key[1:]
                if Ellipsis in key:
                    raise IndexingError(_one_ellipsis_message)
                return self._validate_key_length(key)
            raise IndexingError("Too many indexers")
        return key

    @final
    def _getitem_tuple_same_dim(self, tup: tuple):
        # [FUNCTION DOC]
        # Final. HELPER: index with a tuple of keys that does not reduce dimensions.
        # Only called after the lower-dimension attempt failed. Parameter tup. Applies
        # each non-null key one axis at a time starting from the last axis (columns first
        # since it is faster), asserting the dimension is preserved. If every key was a
        # null slice returns a shallow copy so that the result is a new object.
        """
        Index with indexers that should return an object of the same dimension
        as self.obj.

        This is only called after a failed call to _getitem_lowerdim.
        """
        retval = self.obj
        # Selecting columns before rows is significantly faster
        start_val = (self.ndim - len(tup)) + 1
        for i, key in enumerate(reversed(tup)):
            i = self.ndim - i - start_val
            if com.is_null_slice(key):
                continue

            retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
            # We should never have retval.ndim < self.ndim, as that should
            #  be handled by the _getitem_lowerdim call above.
            assert retval.ndim == self.ndim

        if retval is self.obj:
            # if all axes were a null slice (`df.loc[:, :]`), ensure we still
            # return a new object (https://github.com/pandas-dev/pandas/pull/49469)
            retval = retval.copy(deep=False)

        return retval

    @final
    def _getitem_lowerdim(self, tup: tuple):
        # [FUNCTION DOC]
        # Final. HELPER: try to index with a tuple in a way that reduces dimensions.
        # Parameter tup. With an axis restriction, indexes that axis directly. Nested
        # tuples go to the nested-tuple path. For label indexers on a multi-index first
        # axis with no slices in the key, tries the multi-index first-axis path (ignoring
        # an indexing error). Otherwise validates the length, then looks from the last
        # axis backwards for a label-like or list-like key; the section for that key is
        # taken, and the remaining key is applied recursively with the same indexer
        # name on the section (the key axes are dropped, or replaced with a null slice if
        # the dimension was kept). A null remaining key returns the section directly.
        # Raises an indexing error saying not applicable if nothing matched.
        # we can directly get the axis result since the axis is specified
        if self.axis is not None:
            axis = self.obj._get_axis_number(self.axis)
            return self._getitem_axis(tup, axis=axis)

        # we may have a nested tuples indexer here
        if self._is_nested_tuple_indexer(tup):
            return self._getitem_nested_tuple(tup)

        # we maybe be using a tuple to represent multiple dimensions here
        ax0 = self.obj._get_axis(0)
        # ...but iloc should handle the tuple as simple integer-location
        # instead of checking it as multiindex representation (GH 13797)
        if (
            isinstance(ax0, MultiIndex)
            and self.name != "iloc"
            and not any(isinstance(x, slice) for x in tup)
        ):
            # Note: in all extant test cases, replacing the slice condition with
            #  `all(is_hashable(x) or com.is_null_slice(x) for x in tup)`
            #  is equivalent.
            #  (see the other place where we call _handle_lowerdim_multi_index_axis0)
            with suppress(IndexingError):
                return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)

        tup = self._validate_key_length(tup)

        # Reverse tuple so that we are indexing along columns before rows
        # and avoid unintended dtype inference. # GH60600
        for i, key in zip(range(len(tup) - 1, -1, -1), reversed(tup), strict=True):
            if is_label_like(key) or is_list_like(key):
                # We don't need to check for tuples here because those are
                #  caught by the _is_nested_tuple_indexer check above.
                section = self._getitem_axis(key, axis=i)

                # We should never have a scalar section here, because
                #  _getitem_lowerdim is only called after a check for
                #  is_scalar_access, which that would be.
                if section.ndim == self.ndim:
                    # we're in the middle of slicing through a MultiIndex
                    # revise the key wrt to `section` by inserting an _NS
                    new_key = (*tup[:i], _NS, *tup[i + 1 :])

                else:
                    # Note: the section.ndim == self.ndim check above
                    #  rules out having DataFrame here, so we dont need to worry
                    #  about transposing.
                    new_key = tup[:i] + tup[i + 1 :]

                    if len(new_key) == 1:
                        new_key = new_key[0]

                # Slices should return views, but calling iloc/loc with a null
                # slice returns a new object.
                if com.is_null_slice(new_key):
                    return section
                # This is an elided recursive call to iloc/loc
                return getattr(section, self.name)[new_key]

        raise IndexingError("not applicable")

    @final
    def _getitem_nested_tuple(self, tup: tuple):
        # [FUNCTION DOC]
        # Final. HELPER: handle tuples containing nested tuples (multi-index levels).
        # Parameter tup. Validates each element is not a dict or set. If the tuple is
        # longer than the dimension: only label indexers are allowed (otherwise a value
        # error); all-hashable-or-null-slice tuples try the multi-index first-axis path
        # (ignoring an indexing error); a series with a tuple inside the tuple raises an
        # indexing error; then the whole tuple is applied on the restricted or first axis.
        # Otherwise applies each non-null key from the last axis backwards, stopping
        # as soon as a scalar is reached. Returns the result.
        # we have a nested tuple so have at least 1 multi-index level
        # we should be able to match up the dimensionality here

        for key in tup:
            check_dict_or_set_indexers(key)

        # we have too many indexers for our dim, but have at least 1
        # multi-index dimension, try to see if we have something like
        # a tuple passed to a series with a multi-index
        if len(tup) > self.ndim:
            if self.name != "loc":
                # This should never be reached, but let's be explicit about it
                raise ValueError("Too many indices")  # pragma: no cover
            if all(
                is_hashable(x, allow_slice=False) or com.is_null_slice(x) for x in tup
            ):
                # GH#10521 Series should reduce MultiIndex dimensions instead of
                #  DataFrame, IndexingError is not raised when slice(None,None,None)
                #  with one row.
                with suppress(IndexingError):
                    return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(
                        tup
                    )
            elif isinstance(self.obj, ABCSeries) and any(
                isinstance(k, tuple) for k in tup
            ):
                # GH#35349 Raise if tuple in tuple for series
                # Do this after the all-hashable-or-null-slice check so that
                #  we are only getting non-hashable tuples, in particular ones
                #  that themselves contain a slice entry
                # See test_loc_series_getitem_too_many_dimensions
                raise IndexingError("Too many indexers")

            # this is a series with a multi-index specified a tuple of
            # selectors
            axis = self.axis or 0
            return self._getitem_axis(tup, axis=axis)

        # handle the multi-axis by taking sections and reducing
        # this is iterative
        obj = self.obj
        # GH#41369 Loop in reverse order ensures indexing along columns before rows
        # which selects only necessary blocks which avoids dtype conversion if possible
        axis = len(tup) - 1
        for key in reversed(tup):
            if com.is_null_slice(key):
                axis -= 1
                continue

            obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
            axis -= 1

            # if we have a scalar, we are done
            if is_scalar(obj) or not hasattr(obj, "ndim"):
                break

        return obj

    def _convert_to_indexer(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # Abstract hook: convert a key to a positional indexer for an axis. Parameters:
        # key; axis. Subclasses implement it; the base raises the abstract method error.
        raise AbstractMethodError(self)

    def _raise_callable_usage(self, key: Any, maybe_callable: T) -> T:
        # [FUNCTION DOC]
        # HELPER: if the indexer is position-based and the key was a callable that
        # returned a tuple, raises a value error saying returning a tuple from a callable
        # is not allowed; otherwise returns the evaluated key. Parameters: key (original);
        # maybe_callable (result of applying the callable).
        # GH53533
        if self.name == "iloc" and callable(key) and isinstance(maybe_callable, tuple):
            raise ValueError(
                "Returning a tuple from a callable with iloc is not allowed.",
            )
        return maybe_callable

    @final
    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Final. Get items through the indexer. Parameter key. Validates it is not a
        # dict or set. Tuple keys have iterators converted to lists and callables
        # applied; if the tuple addresses a single scalar the value is read directly
        # from the owner, else the tuple routine of the subclass is used. Non-tuple keys
        # are evaluated if callable (with the tuple check) and looked up on the
        # restricted axis or the first axis.
        check_dict_or_set_indexers(key)
        if type(key) is tuple:
            key = (list(x) if is_iterator(x) else x for x in key)
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
            if self._is_scalar_access(key):
                return self.obj._get_value(*key, takeable=self._takeable)
            return self._getitem_tuple(key)
        else:
            # we by definition only have the 0th axis
            axis = self.axis or 0

            maybe_callable = com.apply_if_callable(key, self.obj)
            maybe_callable = self._raise_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _is_scalar_access(self, key: tuple):
        # [FUNCTION DOC]
        # Abstract hook: whether a tuple key addresses a single scalar element.
        # Parameter key. The base raises a not-implemented error.
        raise NotImplementedError

    def _getitem_tuple(self, tup: tuple):
        # [FUNCTION DOC]
        # Abstract hook: get with a tuple key. Parameter tup. The base raises the
        # abstract method error.
        raise AbstractMethodError(self)

    def _getitem_axis(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # Abstract hook: get with a single key along an axis. Parameters: key; axis. The
        # base raises a not-implemented error.
        raise NotImplementedError

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        # [FUNCTION DOC]
        # Abstract hook: check that a positional indexer is valid for setting. Parameter
        # indexer. The base raises the abstract method error.
        raise AbstractMethodError(self)

    @final
    def _getbool_axis(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # Final. HELPER: select by a boolean key along an axis (the caller guarantees
        # the axis is not None). Aligns/validates the boolean key against the axis labels,
        # finds the true positions and takes them from the owner. Returns the result.
        # caller is responsible for ensuring non-None axis
        labels = self.obj._get_axis(axis)
        key = check_bool_indexer(labels, key)
        inds = key.nonzero()[0]
        return self.obj.take(inds, axis=axis)


@doc(IndexingMixin.loc)
class _LocIndexer(_LocationIndexer):
    # [CLASS DOC]
    # Label-based indexer (the object behind the loc accessor). Subclass of the
    # location-indexer base; its documentation is copied from the loc property.
    # Class-level state: the integer-keys-are-positions flag is false and a string
    # describing the valid key types (labels, label slices with both endpoints,
    # list-likes of labels, booleans). Implements key validation, scalar-access
    # detection, multi-take shortcuts, iterable/slice/boolean/label retrieval and
    # the conversion of label keys to positional indexers.
    _takeable: bool = False
    _valid_types = (
        "labels (MUST BE IN THE INDEX), slices of labels (BOTH "
        "endpoints included! Can be slices of integers if the "
        "index is integers), listlike of labels, boolean"
    )

    # -------------------------------------------------------------------
    # Key Checks

    @doc(_LocationIndexer._validate_key)
    def _validate_key(self, key, axis: Axis) -> None:
        # [FUNCTION DOC]
        # Validate a key for label indexing (docstring inherited from the base hook).
        # Parameters: key; axis. A boolean key is only allowed if the axis has boolean
        # labels (boolean dtype or nullable boolean or multi-index whose first level is
        # boolean); otherwise raises a key error. A slice with a boolean start or stop
        # raises a type error. Other keys are accepted (presence is checked later).
        # valid for a collection of labels (we check their presence later)
        # slice of labels (where start-end in labels)
        # slice of integers (only if in the labels)
        # boolean not in slice and with boolean index
        ax = self.obj._get_axis(axis)
        if isinstance(key, bool) and not (
            is_bool_dtype(ax.dtype)
            or ax.dtype.name == "boolean"
            or (
                isinstance(ax, MultiIndex)
                and is_bool_dtype(ax.get_level_values(0).dtype)
            )
        ):
            raise KeyError(
                f"{key}: boolean label can not be used without a boolean index"
            )

        if isinstance(key, slice) and (
            isinstance(key.start, bool) or isinstance(key.stop, bool)
        ):
            raise TypeError(f"{key}: boolean values can not be used in a slice")

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        # [FUNCTION DOC]
        # Always returns true: label-based setting has no extra positional validity
        # requirements. Parameter indexer.
        return True

    def _is_scalar_access(self, key: tuple) -> bool:
        # [FUNCTION DOC]
        # True if the tuple key addresses a single scalar for the fast scalar path.
        # Parameter key. Requires the length to equal the dimension and every
        # element to be a scalar; false if any axis is a multi-index, if a string key
        # is used on an axis supporting partial-string indexing (e.g. dates), or if an
        # axis labels are not unique.
        """
        Returns
        -------
        bool
        """
        # this is a shortcut accessor to both .loc and .iloc
        # that provide the equivalent access of .at and .iat
        # a) avoid getting things via sections and (to minimize dtype changes)
        # b) provide a performant path
        if len(key) != self.ndim:
            return False

        for i, k in enumerate(key):
            if not is_scalar(k):
                return False

            ax = self.obj.axes[i]
            if isinstance(ax, MultiIndex):
                return False

            if isinstance(k, str) and ax._supports_partial_string_indexing:
                # partial string indexing, df.loc['2000', 'A']
                # should not be considered scalar
                return False

            if not ax._index_as_unique:
                return False

        return True

    # -------------------------------------------------------------------
    # MultiIndex Handling

    def _multi_take_opportunity(self, tup: tuple) -> bool:
        # [FUNCTION DOC]
        # HELPER: true if the multi-take shortcut can be used: every element of the
        # tuple is a list-like indexer and none is a boolean indexer. Parameter tup.
        """
        Check whether there is the possibility to use ``_multi_take``.

        Currently the limit is that all axes being indexed, must be indexed with
        list-likes.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        bool
            Whether the current indexing,
            can be passed through `_multi_take`.
        """
        if not all(is_list_like_indexer(x) for x in tup):
            return False

        # just too complicated
        return not any(com.is_bool_indexer(x) for x in tup)

    def _multi_take(self, tup: tuple):
        # [FUNCTION DOC]
        # HELPER: index all axes at once for a tuple of list-like keys. Parameter tup:
        # one key per axis. Gets the label indexers for each axis, then reindexes the
        # owner with all of them in one operation (duplicates allowed). Returns an object
        # of the same type as the owner.
        """
        Create the indexers for the passed tuple of keys, and
        executes the take operation. This allows the take operation to be
        executed all at once, rather than once for each dimension.
        Improving efficiency.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        values: same type as the object being indexed
        """
        # GH 836
        d = {
            axis: self._get_listlike_indexer(key, axis)
            for (key, axis) in zip(tup, self.obj._AXIS_ORDERS, strict=True)
        }
        return self.obj._reindex_with_indexers(d, allow_dups=True)

    # -------------------------------------------------------------------

    def _getitem_iterable(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # HELPER: index with an iterable of labels (non-boolean) along an axis.
        # Parameters: key; axis. Validates the key, gets the matching labels and
        # positions and reindexes the owner along the axis allowing duplicates. Raises
        # key errors for missing labels (through the lookup). Returns an object like the
        # owner.
        """
        Index current object with an iterable collection of keys.

        Parameters
        ----------
        key : iterable
            Targeted labels.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If no key was found. Will change in the future to raise if not all
            keys were found.

        Returns
        -------
        scalar, DataFrame, or Series: indexed value(s).
        """
        # we assume that not com.is_bool_indexer(key), as that is
        #  handled before we get here.
        self._validate_key(key, axis)

        # A collection of keys
        keyarr, indexer = self._get_listlike_indexer(key, axis)
        return self.obj._reindex_with_indexers(
            {axis: [keyarr, indexer]}, allow_dups=True
        )

    def _getitem_tuple(self, tup: tuple):
        # [FUNCTION DOC]
        # Get with a tuple key. Parameter tup. First tries ellipsis expansion and the
        # lower-dimension path, ignoring an indexing error. If that failed, validates all
        # elements as a tuple of valid label keys, uses the multi-take shortcut when
        # every element is list-like, and otherwise indexes keeping the dimensions.
        with suppress(IndexingError):
            tup = self._expand_ellipsis(tup)
            return self._getitem_lowerdim(tup)

        # no multi-index, so validate all of the indexers
        tup = self._validate_tuple_indexer(tup)

        # ugly hack for GH #836
        if self._multi_take_opportunity(tup):
            return self._multi_take(tup)

        return self._getitem_tuple_same_dim(tup)

    def _get_label(self, label, axis: AxisInt):
        # [FUNCTION DOC]
        # HELPER: fetch the cross-section for a label along an axis, via the owner's
        # cross-section method. Parameters: label; axis. Raises a key error when the
        # label is missing.
        # GH#5567 this will fail if the label is not present in the axis.
        return self.obj.xs(label, axis=axis)

    def _handle_lowerdim_multi_index_axis0(self, tup: tuple):
        # [FUNCTION DOC]
        # HELPER: handle a tuple key on an object whose first axis is a multi-index.
        # Parameter tup. Tries the cross-section for the tuple on the restricted or first
        # axis. On a key error, re-raises it if the number of keys exceeds the
        # dimension but fits in the number of index levels; otherwise raises an
        # indexing error saying no label was returned.
        # we have an axis0 multi-index, handle or raise
        axis = self.axis or 0
        try:
            # fast path for series or for tup devoid of slices
            return self._get_label(tup, axis=axis)

        except KeyError as ek:
            # raise KeyError if number of indexers match
            # else IndexingError will be raised
            if self.ndim < len(tup) <= self.obj.index.nlevels:
                raise ek
            raise IndexingError("No label returned") from ek

    def _getitem_axis(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # Get with a single key along an axis. Parameters: key; axis. Zero-dimensional
        # arrays become scalars, iterators become lists, an ellipsis becomes a full
        # slice. A slice is validated and applied as a label slice; boolean indexers
        # use the boolean path; list-likes (not multi-index tuples) must be at most 1D
        # (else value error) and use the iterable path; nested tuples on a
        # multi-index use the multi-index locations and then position-based indexing;
        # anything else is validated and looked up as a label.
        # KNOWN ISSUE: the conversion of a tuple key to a tuple on a multi-index axis is a no-op because the key is already a tuple.
        key = item_from_zerodim(key)
        if is_iterator(key):
            key = list(key)
        if key is Ellipsis:
            key = slice(None)

        labels = self.obj._get_axis(axis)

        if isinstance(key, tuple) and isinstance(labels, MultiIndex):
            key = tuple(key)

        if isinstance(key, slice):
            self._validate_key(key, axis)
            return self._get_slice_axis(key, axis=axis)
        elif com.is_bool_indexer(key):
            return self._getbool_axis(key, axis=axis)
        elif is_list_like_indexer(key):
            # an iterable multi-selection
            if not (isinstance(key, tuple) and isinstance(labels, MultiIndex)):
                if hasattr(key, "ndim") and key.ndim > 1:
                    raise ValueError("Cannot index with multidimensional key")

                return self._getitem_iterable(key, axis=axis)

            # nested tuple slicing
            if is_nested_tuple(key, labels):
                locs = labels.get_locs(key)
                indexer: list[slice | npt.NDArray[np.intp]] = [slice(None)] * self.ndim
                indexer[axis] = locs
                return self.obj.iloc[tuple(indexer)]

        # fall thru to straight lookup
        self._validate_key(key, axis)
        return self._get_label(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        # [FUNCTION DOC]
        # HELPER: slice by labels along an axis (caller guarantees the axis is set).
        # Parameters: slice_obj (slice of labels); axis. If the slice selects everything
        # a shallow copy is returned. Otherwise the labels convert the slice to a
        # positional slice or, for datetime indexes, an index-like indexer; slices
        # use the owner's slicing, other indexers use take. Returns the result.
        """
        This is pretty simple as we just have to deal with labels.
        """
        # caller is responsible for ensuring non-None axis
        obj = self.obj
        if not need_slice(slice_obj):
            return obj.copy(deep=False)

        labels = obj._get_axis(axis)
        indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop, slice_obj.step)

        if isinstance(indexer, slice):
            return self.obj._slice(indexer, axis=axis)
        else:
            # DatetimeIndex overrides Index.slice_indexer and may
            #  return a DatetimeIndex instead of a slice object.
            return self.obj.take(indexer, axis=axis)

    def _convert_to_indexer(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # Convert a label key into something usable for positional access along an axis
        # (label-based version of the abstract hook). Parameters: key; axis.
        # Slices are converted by the labels as label slices. A tuple with more than one
        # element on a series with a non-multi index raises an indexing error (too many
        # indexers). Scalars, and hashable keys on multi-indexes that contain no slice,
        # are looked up directly; a missing tuple key of full length on a multi-index,
        # and a value error from an integer key, return a one-entry marker mapping
        # holding the key (meaning: missing, to be enlarged when setting); other
        # lookup failures propagate. Nested tuples use multi-level location lookup (a series
        # with a tuple inside raises an indexing error). List-likes: iterators are
        # materialized; boolean keys are validated against the labels and returned;
        # others return the positional indexer of the list lookup. Anything else is
        # looked up, with a missing non-list key returning the missing marker.
        # KNOWN ISSUE: the existing docstring describes an old accessor and an ambiguity error that no longer exist and omits the marker mapping returned for missing keys.
        # KNOWN ISSUE: in the final branch the re-raise after the list-like test is unreachable, because only non-list-like keys reach that branch.
        """
        Convert indexing key into something we can use to do actual fancy
        indexing on an ndarray.

        Examples
        ix[:5] -> slice(0, 5)
        ix[[1,2,3]] -> [1,2,3]
        ix[['foo', 'bar', 'baz']] -> [i, j, k] (indices of foo, bar, baz)

        Going by Zen of Python?
        'In the face of ambiguity, refuse the temptation to guess.'
        raise AmbiguousIndexError with integer labels?
        - No, prefer label-based indexing
        """
        labels = self.obj._get_axis(axis)

        if isinstance(key, slice):
            return labels._convert_slice_indexer(key, kind="loc")

        if (
            isinstance(key, tuple)
            and not isinstance(labels, MultiIndex)
            and self.ndim < 2
            and len(key) > 1
        ):
            raise IndexingError("Too many indexers")

        # Slices are not valid keys passed in by the user,
        # even though they are hashable in Python 3.12
        contains_slice = False
        if isinstance(key, tuple):
            contains_slice = any(isinstance(v, slice) for v in key)

        if is_scalar(key) or (
            isinstance(labels, MultiIndex) and is_hashable(key) and not contains_slice
        ):
            # Otherwise get_loc will raise InvalidIndexError

            # if we are a label return me
            try:
                return labels.get_loc(key)
            except LookupError:
                if isinstance(key, tuple) and isinstance(labels, MultiIndex):
                    if len(key) == labels.nlevels:
                        return {"key": key}
                    raise
            except InvalidIndexError:
                # GH35015, using datetime as column indices raises exception
                if not isinstance(labels, MultiIndex):
                    raise
            except ValueError:
                if not is_integer(key):
                    raise
                return {"key": key}

        if is_nested_tuple(key, labels):
            if self.ndim == 1 and any(isinstance(k, tuple) for k in key):
                # GH#35349 Raise if tuple in tuple for series
                raise IndexingError("Too many indexers")
            return labels.get_locs(key)

        elif is_list_like_indexer(key):
            if is_iterator(key):
                key = list(key)

            if com.is_bool_indexer(key):
                key = check_bool_indexer(labels, key)
                return key
            else:
                return self._get_listlike_indexer(key, axis)[1]
        else:
            try:
                return labels.get_loc(key)
            except LookupError:
                # allow a not found key only if we are a setter
                if not is_list_like_indexer(key):
                    return {"key": key}
                raise

    def _get_listlike_indexer(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # HELPER: transform a list-like of labels into an index and a positional indexer.
        # Parameters: key; axis. Uses the axis labels' strict indexer routine with the axis
        # name. Raises a key error if keys were requested but none found (or any are
        # missing, per the strict routine). Returns a pair of (the new index, the
        # indexer array with minus one for missing labels).
        """
        Transform a list-like of keys into a new index and an indexer.

        Parameters
        ----------
        key : list-like
            Targeted labels.
        axis:  int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.

        Returns
        -------
        keyarr: Index
            New index (coinciding with 'key' if the axis is unique).
        values : array-like
            Indexer for the return object, -1 denotes keys not found.
        """
        ax = self.obj._get_axis(axis)
        axis_name = self.obj._get_axis_name(axis)

        keyarr, indexer = ax._get_indexer_strict(key, axis_name)

        return keyarr, indexer


@doc(IndexingMixin.iloc)
class _iLocIndexer(_LocationIndexer):
    # [CLASS DOC]
    # Position-based indexer (the object behind the iloc accessor). Subclass of the
    # location-indexer base; its documentation is copied from the iloc property.
    # Class-level state: valid-types description (integer, integer slice, list-like of
    # integers, boolean array); integer keys are positions (true). Implements key
    # validation, scalar-access detection, positional retrieval (slice, list,
    # boolean, integer), the identity key conversion, and the whole setting machinery
    # (single block, split by column, frame/2D values, new column or row creation,
    # series alignment).
    _valid_types = (
        "integer, integer slice (START point is INCLUDED, END "
        "point is EXCLUDED), listlike of integers, boolean array"
    )
    _takeable = True

    # -------------------------------------------------------------------
    # Key Checks

    def _validate_key(self, key, axis: AxisInt) -> None:
        # [FUNCTION DOC]
        # Validate a key for position indexing along an axis. Parameters: key; axis.
        # Boolean indexers pass (if the key has a pandas index it must be integer typed,
        # else a value error says an indexable cannot be used as a mask). Slices pass.
        # Integers are range-checked. A tuple raises an indexing error. List-likes must
        # be numeric (else an index error) and within bounds (else an index error that
        # positional indexers are out of bounds); empty lists pass. Any other key raises a
        # value error listing the valid types.
        if com.is_bool_indexer(key):
            if hasattr(key, "index") and isinstance(key.index, Index):
                if key.index.inferred_type == "integer":
                    return
                raise ValueError(
                    "iLocation based boolean indexing cannot use an indexable as a mask"
                )
            return

        if isinstance(key, slice):
            return
        elif is_integer(key):
            self._validate_integer(key, axis)
        elif isinstance(key, tuple):
            # a tuple should already have been caught by this point
            # so don't treat a tuple as a valid indexer
            raise IndexingError("Too many indexers")
        elif is_list_like_indexer(key):
            if isinstance(key, ABCSeries):
                arr = key._values
            elif is_array_like(key):
                arr = key
            else:
                arr = np.array(key)
            len_axis = len(self.obj._get_axis(axis))

            # check that the key has a numeric dtype
            if not is_numeric_dtype(arr.dtype):
                raise IndexError(f".iloc requires numeric indexers, got {arr}")

            if len(arr):
                if isinstance(arr.dtype, ExtensionDtype):
                    arr_max = arr._reduce("max")
                    arr_min = arr._reduce("min")
                else:
                    arr_max = np.max(arr)
                    arr_min = np.min(arr)

                # check that the key does not exceed the maximum size
                if arr_max >= len_axis or arr_min < -len_axis:
                    raise IndexError("positional indexers are out-of-bounds")
        else:
            raise ValueError(f"Can only index by location with a [{self._valid_types}]")

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        # [FUNCTION DOC]
        # Validate that a positional indexer cannot enlarge the object (does not modify
        # it). Parameter indexer. A mapping (missing marker) raises an index error saying
        # iloc cannot enlarge its target; a frame indexer raises a type error suggesting
        # the label indexer. A non-tuple is expanded to a tuple; then each element is
        # checked: integers beyond the axis length or mapping elements raise the same
        # index error, slices and list-likes are not checked. Returns true.
        """
        Validate that a positional indexer cannot enlarge its target
        will raise if needed, does not modify the indexer externally.

        Returns
        -------
        bool
        """
        if isinstance(indexer, dict):
            raise IndexError("iloc cannot enlarge its target object")

        if isinstance(indexer, ABCDataFrame):
            raise TypeError(
                "DataFrame indexer for .iloc is not supported. "
                "Consider using .loc with a DataFrame indexer for automatic alignment.",
            )

        if not isinstance(indexer, tuple):
            indexer = _tuplify(self.ndim, indexer)

        for ax, i in zip(self.obj.axes, indexer, strict=False):
            if isinstance(i, slice):
                # should check the stop slice?
                pass
            elif is_list_like_indexer(i):
                # should check the elements?
                pass
            elif is_integer(i):
                if i >= len(ax):
                    raise IndexError("iloc cannot enlarge its target object")
            elif isinstance(i, dict):
                raise IndexError("iloc cannot enlarge its target object")

        return True

    def _is_scalar_access(self, key: tuple) -> bool:
        # [FUNCTION DOC]
        # True if the tuple key addresses a single scalar: its length equals the
        # dimension and every element is an integer. Parameter key.
        """
        Returns
        -------
        bool
        """
        # this is a shortcut accessor to both .loc and .iloc
        # that provide the equivalent access of .at and .iat
        # a) avoid getting things via sections and (to minimize dtype changes)
        # b) provide a performant path
        if len(key) != self.ndim:
            return False

        return all(is_integer(k) for k in key)

    def _validate_integer(self, key: int | np.integer, axis: AxisInt) -> None:
        # [FUNCTION DOC]
        # HELPER: check that an integer is a valid position on an axis. Parameters: key
        # (integer); axis. Raises an index error saying a single positional indexer is out
        # of bounds if the key is at least the axis length or below minus the length.
        """
        Check that 'key' is a valid position in the desired axis.

        Parameters
        ----------
        key : int
            Requested position.
        axis : int
            Desired axis.

        Raises
        ------
        IndexError
            If 'key' is not a valid position in axis 'axis'.
        """
        len_axis = len(self.obj._get_axis(axis))
        if key >= len_axis or key < -len_axis:
            raise IndexError("single positional indexer is out-of-bounds")

    # -------------------------------------------------------------------

    def _getitem_tuple(self, tup: tuple):
        # [FUNCTION DOC]
        # Get with a tuple key. Parameter tup. Validates the tuple, tries the
        # lower-dimension path (ignoring an indexing error), and otherwise indexes keeping
        # the dimensions.
        tup = self._validate_tuple_indexer(tup)
        with suppress(IndexingError):
            return self._getitem_lowerdim(tup)

        return self._getitem_tuple_same_dim(tup)

    def _get_list_axis(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # HELPER: select positions given by a list or array of integers along an axis.
        # Parameters: key; axis. Takes those positions from the owner; an index error from
        # the take is re-raised with a message that positional indexers are out of
        # bounds. Returns an object like the owner.
        # KNOWN ISSUE: the existing docstring says it returns series values and that the axis can only be zero, but the code works along any axis and returns whatever type the owner is.
        """
        Return Series values by list or array of integers.

        Parameters
        ----------
        key : list-like positional indexer
        axis : int

        Returns
        -------
        Series object

        Notes
        -----
        `axis` can only be zero.
        """
        try:
            return self.obj.take(key, axis=axis)
        except IndexError as err:
            # re-raise with different error message, e.g. test_getitem_ndarray_3d
            raise IndexError("positional indexers are out-of-bounds") from err

    def _getitem_axis(self, key, axis: AxisInt):
        # [FUNCTION DOC]
        # Get with a single key along an axis. Parameters: key; axis. An ellipsis
        # becomes a full slice; a frame key raises an index error suggesting the label
        # indexer. Slices use the slice path; iterators and lists become arrays; boolean
        # indexers are validated and applied; list-likes of integers use the list path;
        # otherwise the key (zero-dimension arrays unwrapped) must be an integer (else
        # a type error), is range-checked and fetched with the owner's position fetch.
        if key is Ellipsis:
            key = slice(None)
        elif isinstance(key, ABCDataFrame):
            raise IndexError(
                "DataFrame indexer is not allowed for .iloc\n"
                "Consider using .loc for automatic alignment."
            )

        if isinstance(key, slice):
            return self._get_slice_axis(key, axis=axis)

        if is_iterator(key):
            key = list(key)

        if isinstance(key, list):
            key = np.asarray(key)

        if com.is_bool_indexer(key):
            self._validate_key(key, axis)
            return self._getbool_axis(key, axis=axis)

        # a list of integers
        elif is_list_like_indexer(key):
            return self._get_list_axis(key, axis=axis)

        # a single integer
        else:
            key = item_from_zerodim(key)
            if not is_integer(key):
                raise TypeError("Cannot index by location index with a non-integer key")

            # validate the location
            self._validate_integer(key, axis)

            return self.obj._ixs(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        # [FUNCTION DOC]
        # HELPER: slice by position along an axis (the caller guarantees the axis is
        # set). Parameters: slice_obj; axis. A slice that selects everything returns a
        # shallow copy; otherwise the axis labels validate the slice as positional and
        # the owner slices it.
        # caller is responsible for ensuring non-None axis
        obj = self.obj

        if not need_slice(slice_obj):
            return obj.copy(deep=False)

        labels = obj._get_axis(axis)
        labels._validate_positional_slice(slice_obj)
        return self.obj._slice(slice_obj, axis=axis)

    def _convert_to_indexer(self, key: T, axis: AxisInt) -> T:
        # [FUNCTION DOC]
        # Identity conversion: position keys are already positional. Parameters: key;
        # axis (unused). Returns the key unchanged.
        """
        Much simpler as we only have to deal with our valid types.
        """
        return key

    def _get_setitem_indexer(self, key):
        # [FUNCTION DOC]
        # Convert a key for setting (position version: no conversion). Parameter key.
        # Iterators are turned into lists (numpy validates later); with an axis
        # restriction the key is expanded to a full tuple. Returns the key.
        # GH#32257 Fall through to let numpy do validation
        if is_iterator(key):
            key = list(key)

        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)

        return key

    # -------------------------------------------------------------------

    def _decide_split_path(self, indexer, value) -> bool:
        # [FUNCTION DOC]
        # HELPER: decide whether to set column by column (the split path) or in one block.
        # Parameters: indexer; value. Split if the owner has more than one block; if the
        # value is a frame with more than one block; if the single block cannot hold the
        # value (for a 2D object; dict values use their values); or if the indexer is a
        # full-length tuple with a multi-index axis whose element is neither an integer
        # nor a null slice. Returns a boolean.
        """
        Decide whether we will take a block-by-block path.
        """
        take_split_path = not self.obj._mgr.is_single_block

        if not take_split_path and isinstance(value, ABCDataFrame):
            # Avoid cast of values
            take_split_path = not value._mgr.is_single_block

        # if there is only one block/type, still have to take split path
        # unless the block is one-dimensional or it can hold the value
        if not take_split_path and len(self.obj._mgr.blocks) and self.ndim > 1:
            # in case of dict, keys are indices
            val = list(value.values()) if isinstance(value, dict) else value
            arr = self.obj._mgr.blocks[0].values
            take_split_path = not can_hold_element(
                arr, extract_array(val, extract_numpy=True)
            )

        # if we have any multi-indexes that have non-trivial slices
        # (not null slices) then we must take the split path, xref
        # GH 10360, GH 27841
        if isinstance(indexer, tuple) and len(indexer) == len(self.obj.axes):
            for i, ax in zip(indexer, self.obj.axes, strict=True):
                if isinstance(ax, MultiIndex) and not (
                    is_integer(i) or com.is_null_slice(i)
                ):
                    take_split_path = True
                    break

        return take_split_path

    def _setitem_new_column(self, indexer, key, value, name: str) -> None:
        # [FUNCTION DOC]
        # HELPER: handle setting with a missing column label by adding the column.
        # Parameters: indexer (tuple with missing marker); key (the new column label);
        # value; name (indexer name). On an empty object a non-list-like value raises a
        # value error (cannot set a frame with no defined index and a scalar), and
        # otherwise the column is assigned with the frame setitem. A null-slice row
        # indexer assigns the entire column. An array-like value builds an all-missing
        # column of the right type, fills the selected rows directly unless a series (which
        # needs alignment); a scalar creates a column whose dtype is inferred from the
        # value; other list-likes infer a fill column (flagged in a comment as a known
        # unresolved case). Finally converts the indexer to positional form and sets the
        # values through the main path. Returns nothing.
        """
        _setitem_with_indexer cases that can go through DataFrame.__setitem__.
        """
        # add the new item, and set the value
        # must have all defined axes if we have a scalar
        # or a list-like on the non-info axes if we have a
        # list-like
        if not len(self.obj):
            if not is_list_like_indexer(value):
                raise ValueError(
                    "cannot set a frame with no defined index and a scalar"
                )
            self.obj[key] = value
            return

        # add a new item with the dtype setup
        if com.is_null_slice(indexer[0]):
            # We are setting an entire column
            self.obj[key] = value
            return
        elif is_array_like(value):
            # GH#42099
            arr = extract_array(value, extract_numpy=True)
            taker = -1 * np.ones(len(self.obj), dtype=np.intp)
            empty_value = algos.take_nd(arr, taker)
            if not isinstance(value, ABCSeries):
                # if not Series (in which case we need to align),
                #  we can short-circuit
                if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == 1:
                    # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
                    arr = arr[0, ...]
                empty_value[indexer[0]] = arr
                self.obj[key] = empty_value
                return

            self.obj[key] = empty_value
        elif not is_list_like(value):
            self.obj[key] = construct_1d_array_from_inferred_fill_value(
                value, len(self.obj)
            )
        else:
            # FIXME: GH#42099#issuecomment-864326014
            self.obj[key] = infer_fill_value(value)

        new_indexer = convert_from_missing_indexer_tuple(indexer, self.obj.axes)
        self._setitem_with_indexer(new_indexer, value, name)

        return

    def _setitem_with_indexer(self, indexer, value, name: str = "iloc") -> None:
        # [FUNCTION DOC]
        # Set values on the owner with a positional indexer; may enlarge it.
        # Parameters: indexer (positional or with missing markers); value; name (the
        # indexer name, default iloc).
        # Decides on the split path first. For tuple indexers, each element with a
        # missing marker is handled: on the column axis of a frame the new column route
        # is taken and the method returns; otherwise the axis is extended with the new
        # label (via reindexing with a missing slot) and the element becomes the new
        # position. For non-tuple indexers, a missing marker routes to the row/entry
        # insertion path and returns. For label indexers, applies the mask-value
        # alignment. Then sets column by column (split path) or in a single block.
        """
        _setitem_with_indexer is for setting values on a Series/DataFrame
        using positional indexers.

        If the relevant keys are not present, the Series/DataFrame may be
        expanded.
        """
        info_axis = self.obj._info_axis_number
        take_split_path = self._decide_split_path(indexer, value)

        if isinstance(indexer, tuple):
            nindexer = []
            for i, idx in enumerate(indexer):
                idx, missing = convert_missing_indexer(idx)
                if missing:
                    # reindex the axis to the new value
                    # and set inplace
                    key = idx

                    # if this is the items axes, then take the main missing
                    # path first
                    # this correctly sets the dtype
                    # essentially this separates out the block that is needed
                    # to possibly be modified
                    if self.ndim > 1 and i == info_axis:
                        self._setitem_new_column(indexer, key, value, name=name)
                        return

                    # reindex the axis
                    index = self.obj._get_axis(i)
                    labels = index.insert(len(index), key)

                    # We are expanding the Series/DataFrame values to match
                    #  the length of the new index `labels`.  GH#40096 ensure
                    #  this is valid even if the index has duplicates.
                    taker = np.arange(len(index) + 1, dtype=np.intp)
                    taker[-1] = -1
                    reindexers = {i: (labels, taker)}
                    new_obj = self.obj._reindex_with_indexers(
                        reindexers, allow_dups=True
                    )
                    self.obj._mgr = new_obj._mgr

                    nindexer.append(labels.get_loc(key))

                else:
                    nindexer.append(idx)

            indexer = tuple(nindexer)
        else:
            indexer, missing = convert_missing_indexer(indexer)

            if missing:
                self._setitem_with_indexer_missing(indexer, value)
                return

        if name == "loc":
            # must come after setting of missing
            indexer, value = self._maybe_mask_setitem_value(indexer, value)

        # align and set the values
        if take_split_path:
            # We have to operate column-wise
            self._setitem_with_indexer_split_path(indexer, value, name)
        else:
            self._setitem_single_block(indexer, value, name)

    def _setitem_with_indexer_split_path(self, indexer, value, name: str):
        # [FUNCTION DOC]
        # HELPER: set column by column. Parameters: indexer; value; name. Frame only
        # (asserted). Expands a non-tuple indexer; raises an index error for too many
        # indices and a value error for an array row indexer with more than two
        # dimensions. Series (label indexers) and dicts are aligned to the indexer.
        # Gets the list of column positions and the expected row count. List-like
        # non-scalar values: a frame value goes to the frame path; a 2D value goes to
        # the 2D path; one column with matching length sets all rows of that column; one
        # column with a mismatched nonzero length either retries with a single element (when
        # the value has one element and the column indexer is not an integer) or raises a
        # value error (equal lengths required); an all-false mask with a full-length value is
        # a no-op; scalar-position access to an object column sets nested data; as
        # many values as columns sets one value per column; an empty frame with a full
        # row slice sets the single column (enlargement); anything else raises the same
        # value error. Scalars are set in each selected column.
        """
        Setitem column-wise.
        """
        # Above we only set take_split_path to True for 2D cases
        assert self.ndim == 2

        if not isinstance(indexer, tuple):
            indexer = _tuplify(self.ndim, indexer)
        if len(indexer) > self.ndim:
            raise IndexError("too many indices for array")
        if isinstance(indexer[0], np.ndarray) and indexer[0].ndim > 2:
            raise ValueError(r"Cannot set values with ndim > 2")

        if (isinstance(value, ABCSeries) and name != "iloc") or isinstance(value, dict):
            from pandas import Series

            value = self._align_series(indexer, Series(value))

        # Ensure we have something we can iterate over
        info_axis = indexer[1]
        ilocs = self._ensure_iterable_column_indexer(info_axis)

        pi = indexer[0]
        lplane_indexer = length_of_indexer(pi, self.obj.index)
        # lplane_indexer gives the expected length of obj[indexer[0]]

        # we need an iterable, with an ndim of at least 1
        # eg. don't pass through np.array(0)
        if is_list_like_indexer(value) and getattr(value, "ndim", 1) > 0:
            if isinstance(value, ABCDataFrame):
                self._setitem_with_indexer_frame_value(indexer, value, name)

            elif np.ndim(value) == 2:
                # TODO: avoid np.ndim call in case it isn't an ndarray, since
                #  that will construct an ndarray, which will be wasteful
                self._setitem_with_indexer_2d_value(indexer, value)

            elif len(ilocs) == 1 and lplane_indexer == len(value) and not is_scalar(pi):
                # We are setting multiple rows in a single column.
                self._setitem_single_column(ilocs[0], value, pi)

            elif len(ilocs) == 1 and 0 != lplane_indexer != len(value):
                # We are trying to set N values into M entries of a single
                #  column, which is invalid for N != M
                # Exclude zero-len for e.g. boolean masking that is all-false

                if len(value) == 1 and not is_integer(info_axis):
                    # This is a case like df.iloc[:3, [1]] = [0]
                    #  where we treat as df.iloc[:3, 1] = 0
                    return self._setitem_with_indexer((pi, info_axis[0]), value[0])

                raise ValueError(
                    "Must have equal len keys and value when setting with an iterable"
                )

            elif lplane_indexer == 0 and len(value) == len(self.obj.index):
                # We get here in one case via .loc with an all-False mask
                pass

            elif self._is_scalar_access(indexer) and is_object_dtype(
                self.obj.dtypes._values[ilocs[0]]
            ):
                # We are setting nested data, only possible for object dtype data
                self._setitem_single_column(indexer[1], value, pi)

            elif len(ilocs) == len(value):
                # We are setting multiple columns in a single row.
                for loc, v in zip(ilocs, value, strict=True):
                    self._setitem_single_column(loc, v, pi)

            elif len(ilocs) == 1 and com.is_null_slice(pi) and len(self.obj) == 0:
                # This is a setitem-with-expansion, see
                #  test_loc_setitem_empty_append_expands_rows_mixed_dtype
                # e.g. df = DataFrame(columns=["x", "y"])
                #  df["x"] = df["x"].astype(np.int64)
                #  df.loc[:, "x"] = [1, 2, 3]
                self._setitem_single_column(ilocs[0], value, pi)

            else:
                raise ValueError(
                    "Must have equal len keys and value when setting with an iterable"
                )

        else:
            # scalar value
            for loc in ilocs:
                self._setitem_single_column(loc, value, pi)

    def _setitem_with_indexer_2d_value(self, indexer, value) -> None:
        # [FUNCTION DOC]
        # HELPER: set from a 2D (non-frame) value, column by column. Parameters:
        # indexer; value (list or array of two dimensions). Lists are turned into object
        # arrays. Raises a value error if the number of columns differs from the
        # value's second dimension. Each value column is set in the matching column;
        # object-dtype columns are passed as lists so that types are inferred.
        # We get here with np.ndim(value) == 2, excluding DataFrame,
        #  which goes through _setitem_with_indexer_frame_value
        pi = indexer[0]

        ilocs = self._ensure_iterable_column_indexer(indexer[1])

        if not is_array_like(value):
            # cast lists to array
            value = np.array(value, dtype=object)
        if len(ilocs) != value.shape[1]:
            raise ValueError(
                "Must have equal len keys and value when setting with an ndarray"
            )

        for i, loc in enumerate(ilocs):
            value_col = value[:, i]
            if is_object_dtype(value_col.dtype):
                # casting to list so that we do type inference in setitem_single_column
                value_col = value_col.tolist()
            self._setitem_single_column(loc, value_col, pi)

    def _setitem_with_indexer_frame_value(
        self, indexer, value: DataFrame, name: str
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: set from a frame value column by column. Parameters: indexer; value
        # (DataFrame); name. Position indexer (iloc) sets by position without aligning.
        # Otherwise: with duplicate columns that equal the target's, aligns each target
        # column to the matching value column (missing columns get not-a-number); with other
        # duplicate columns raises a value error (non-unique columns not allowed);
        # with unique columns aligns each by label to the matching value column, missing
        # ones get not-a-number.
        ilocs = self._ensure_iterable_column_indexer(indexer[1])

        sub_indexer = list(indexer)
        pi = indexer[0]

        multiindex_indexer = isinstance(self.obj.columns, MultiIndex)

        unique_cols = value.columns.is_unique

        # We do not want to align the value in case of iloc GH#37728
        if name == "iloc":
            for i, loc in enumerate(ilocs):
                val = value.iloc[:, i]
                self._setitem_single_column(loc, val, pi)

        elif not unique_cols and value.columns.equals(self.obj.columns):
            # We assume we are already aligned, see
            # test_iloc_setitem_frame_duplicate_columns_multiple_blocks
            for loc in ilocs:
                item = self.obj.columns[loc]
                if item in value:
                    sub_indexer[1] = item
                    val = self._align_series(
                        tuple(sub_indexer),
                        value.iloc[:, loc],
                        multiindex_indexer,
                    )
                else:
                    val = np.nan

                self._setitem_single_column(loc, val, pi)

        elif not unique_cols:
            raise ValueError("Setting with non-unique columns is not allowed.")

        else:
            for loc in ilocs:
                item = self.obj.columns[loc]
                if item in value:
                    sub_indexer[1] = item
                    val = self._align_series(
                        tuple(sub_indexer),
                        value[item],
                        multiindex_indexer,
                        using_cow=True,
                    )
                else:
                    val = np.nan

                self._setitem_single_column(loc, val, pi)

    def _setitem_single_column(self, loc: int, value, plane_indexer) -> None:
        # [FUNCTION DOC]
        # HELPER: set values in a single column. Parameters: loc (column position);
        # plane_indexer (integer, slice or list of positions along the rows); value.
        # Nothing is done for empty row selections. A full-column set tries an in-place
        # set first; if that fails with a value, type or lossy-set error, then for
        # non-void, non-object dtypes on a non-empty frame a type error (invalid value
        # for dtype) is raised, else the column is replaced wholesale with the value.
        # Partial sets first replace temporary void-dtype (expansion) columns with a
        # column inferred from the value, then set through the manager.
        # KNOWN ISSUE: the existing docstring has no summary line and lists only two of the three parameters.
        """

        Parameters
        ----------
        loc : int
            Indexer for column position
        plane_indexer : int, slice, listlike[int]
            The indexer we use for setitem along axis=0.
        """
        pi = plane_indexer

        is_full_setter = com.is_null_slice(pi) or com.is_full_slice(pi, len(self.obj))

        is_null_setter = com.is_empty_slice(pi) or (is_array_like(pi) and len(pi) == 0)

        if is_null_setter:
            # no-op, don't cast dtype later
            return

        elif is_full_setter:
            try:
                self.obj._mgr.column_setitem(
                    loc, plane_indexer, value, inplace_only=True
                )
            except (ValueError, TypeError, LossySetitemError) as exc:
                # If we're setting an entire column and we can't do it inplace,
                #  then we can use value's dtype (or inferred dtype)
                #  instead of object
                dtype = self.obj.dtypes.iloc[loc]
                if dtype not in (np.void, object) and not self.obj.empty:
                    # - Exclude np.void, as that is a special case for expansion.
                    #   We want to raise for
                    #       df = pd.DataFrame({'a': [1, 2]})
                    #       df.loc[:, 'a'] = .3
                    #   but not for
                    #       df = pd.DataFrame({'a': [1, 2]})
                    #       df.loc[:, 'b'] = .3
                    # - Exclude `object`, as then no upcasting happens.
                    # - Exclude empty initial object with enlargement,
                    #   as then there's nothing to be inconsistent with.
                    raise TypeError(
                        f"Invalid value '{value}' for dtype '{dtype}'"
                    ) from exc
                self.obj.isetitem(loc, value)
        else:
            # set value into the column (first attempting to operate inplace, then
            #  falling back to casting if necessary)
            dtype = self.obj.dtypes.iloc[loc]
            if dtype == np.void:
                # This means we're expanding, with multiple columns, e.g.
                #     df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
                #     df.loc[df.index <= 2, ['F', 'G']] = (1, 'abc')
                # Columns F and G will initially be set to np.void.
                # Here, we replace those temporary `np.void` columns with
                # columns of the appropriate dtype, based on `value`.
                self.obj.iloc[:, loc] = construct_1d_array_from_inferred_fill_value(
                    value, len(self.obj)
                )
            self.obj._mgr.column_setitem(loc, plane_indexer, value)

    def _setitem_single_block(self, indexer, value, name: str) -> None:
        # [FUNCTION DOC]
        # HELPER: set values when the owner has a single block. Parameters: indexer;
        # value; name. Series (label indexers) and dict values are aligned to the
        # indexer. For a frame tuple indexer with a null-slice row and integer column on
        # a column label that is unique, sets that single column. Other tuple indexers are
        # converted to open-mesh form. Frame values on label indexers are aligned. Finally
        # sets through the manager and replaces the owner's manager.
        """
        _setitem_with_indexer for the case when we have a single Block.
        """
        from pandas import Series

        if (isinstance(value, ABCSeries) and name != "iloc") or isinstance(value, dict):
            # TODO(EA): ExtensionBlock.setitem this causes issues with
            # setting for extensionarrays that store dicts. Need to decide
            # if it's worth supporting that.
            value = self._align_series(indexer, Series(value))

        info_axis = self.obj._info_axis_number
        item_labels = self.obj._get_axis(info_axis)
        if isinstance(indexer, tuple):
            # if we are setting on the info axis ONLY
            # set using those methods to avoid block-splitting
            # logic here
            if (
                self.ndim == len(indexer) == 2
                and is_integer(indexer[1])
                and com.is_null_slice(indexer[0])
            ):
                col = item_labels[indexer[info_axis]]
                if len(item_labels.get_indexer_for([col])) == 1:
                    # e.g. test_loc_setitem_empty_append_expands_rows
                    loc = item_labels.get_loc(col)
                    self._setitem_single_column(loc, value, indexer[0])
                    return

            indexer = maybe_convert_ix(*indexer)  # e.g. test_setitem_frame_align

        if isinstance(value, ABCDataFrame) and name != "iloc":
            value = self._align_frame(indexer, value)._values

        # actually do the set
        self.obj._mgr = self.obj._mgr.setitem(indexer=indexer, value=value)

    def _setitem_with_indexer_missing(self, indexer, value):
        # [FUNCTION DOC]
        # HELPER: enlarge the owner by a new row (frame) or entry (series) while setting.
        # Parameters: indexer (the new label); value.
        # Series: the new label is appended to the index; if the index is unique and the
        # label coerces to an existing label (e.g. a float equal to an integer label)
        # that existing position is set instead (label-based path). Otherwise the
        # new element's dtype is chosen: no forced dtype for non-scalars or missing
        # values that need none; a valid missing marker for the dtype is normalized and
        # the dtype is promoted for it; non-object, non-empty series promote the
        # current dtype for the value; the old values are concatenated with the new
        # element (in a way that handles casting correctly) and the owner's manager is
        # replaced by that of a new series with the new index and the same name.
        # Frame: raises a value error if the frame has no columns. The value is made a
        # series indexed by the columns and named by the new label (series are reindexed;
        # dicts become object series; list-likes must match the number of columns else a
        # value error says mismatched columns). If the frame has no rows, the one-row
        # frame replaces the manager (the index keeps its name or names; dtypes are
        # inferred unless the value already carried a dtype). Otherwise the row is appended
        # with the internal append and its manager replaces the owner's.
        """
        Insert new row(s) or column(s) into the Series or DataFrame.
        """
        from pandas import Series

        # reindex the axis to the new value
        # and set inplace
        if self.ndim == 1:
            index = self.obj.index
            new_index = index.insert(len(index), indexer)

            # we have a coerced indexer, e.g. a float
            # that matches in an int64 Index, so
            # we will not create a duplicate index, rather
            # index to that element
            # e.g. 0.0 -> 0
            # GH#12246
            if index.is_unique:
                # pass new_index[-1:] instead if [new_index[-1]]
                #  so that we retain dtype
                new_indexer = index.get_indexer(new_index[-1:])
                if (new_indexer != -1).any():
                    # We get only here with loc, so can hard code
                    return self._setitem_with_indexer(new_indexer, value, "loc")

            # this preserves dtype of the value and of the object
            if not is_scalar(value):
                new_dtype = None

            elif is_valid_na_for_dtype(value, self.obj.dtype):
                if not is_object_dtype(self.obj.dtype):
                    # Every NA value is suitable for object, no conversion needed
                    value = na_value_for_dtype(self.obj.dtype, compat=False)

                new_dtype = maybe_promote(self.obj.dtype, value)[0]

            elif isna(value):
                new_dtype = None
            elif not self.obj.empty and not is_object_dtype(self.obj.dtype):
                # We should not cast, if we have object dtype because we can
                # set timedeltas into object series
                curr_dtype = self.obj.dtype
                curr_dtype = getattr(curr_dtype, "numpy_dtype", curr_dtype)
                new_dtype = maybe_promote(curr_dtype, value)[0]
            else:
                new_dtype = None

            new_values = Series([value], dtype=new_dtype)._values

            if len(self.obj._values):
                # GH#22717 handle casting compatibility that np.concatenate
                #  does incorrectly
                new_values = concat_compat([self.obj._values, new_values])
            self.obj._mgr = self.obj._constructor(
                new_values, index=new_index, name=self.obj.name
            )._mgr

        elif self.ndim == 2:
            if not len(self.obj.columns):
                # no columns and scalar
                raise ValueError("cannot set a frame with no defined columns")

            has_dtype = hasattr(value, "dtype")
            if isinstance(value, ABCSeries):
                # append a Series
                value = value.reindex(index=self.obj.columns)
                value.name = indexer
            elif isinstance(value, dict):
                value = Series(
                    value, index=self.obj.columns, name=indexer, dtype=object
                )
            else:
                # a list-list
                if is_list_like_indexer(value):
                    # must have conforming columns
                    if len(value) != len(self.obj.columns):
                        raise ValueError("cannot set a row with mismatched columns")

                value = Series(value, index=self.obj.columns, name=indexer)

            if not len(self.obj):
                # We will ignore the existing dtypes instead of using
                #  internals.concat logic
                df = value.to_frame().T

                idx = self.obj.index
                if isinstance(idx, MultiIndex):
                    name = idx.names
                else:
                    name = idx.name

                df.index = Index([indexer], name=name)
                if not has_dtype:
                    # i.e. if we already had a Series or ndarray, keep that
                    #  dtype.  But if we had a list or dict, then do inference
                    df = df.infer_objects()
                self.obj._mgr = df._mgr
            else:
                self.obj._mgr = self.obj._append_internal(value)._mgr

    def _ensure_iterable_column_indexer(self, column_indexer):
        # [FUNCTION DOC]
        # HELPER: make a column indexer iterable. Parameter column_indexer: integer
        # (becomes a one-element list), slice (expanded over the column positions),
        # boolean mask (positions of true entries) or anything else (returned as is).
        # Returns an iterable of column positions.
        """
        Ensure that our column indexer is something that can be iterated over.
        """
        ilocs: Sequence[int | np.integer] | np.ndarray | range
        if is_integer(column_indexer):
            ilocs = [column_indexer]
        elif isinstance(column_indexer, slice):
            ilocs = range(len(self.obj.columns))[column_indexer]
        elif com.is_bool_indexer(column_indexer):
            ilocs = np.arange(len(column_indexer))[
                np.asarray(column_indexer, dtype=bool)
            ]
        else:
            ilocs = column_indexer
        return ilocs

    def _align_series(
        self,
        indexer,
        ser: Series,
        multiindex_indexer: bool = False,
        using_cow: bool = False,
    ):
        # [FUNCTION DOC]
        # HELPER: align a series to the locations that an indexer selects, so it can be
        # assigned. Parameters: indexer (tuple, slice, array, list, index, or integer);
        # ser (series to assign); multiindex_indexer (default false; true if the indexer
        # came from a multi-index, to avoid broadcasting); using_cow (default false;
        # when true a case may return the series without copying).
        # Slices, arrays and lists are treated as one-element tuples; a boolean array in
        # the second place becomes positions. Array elements are flattened. If all
        # elements are sequences or slices the series is reindexed to the selected rows
        # and for several indexers tiled across the selected columns (unless multi-index).
        # Otherwise per axis: for a sequence or slice, the selected labels become an
        # index and the series is returned as a copy of its values (or the series itself
        # when allowed) if empty or already equal, else reindexed to those labels; for a
        # single aligner on a frame with a null slice, the series is reindexed to the columns
        # axis. An integer indexer on a series returns the series for object dtype, a copy
        # of the values if the index is equal, else the reindexed value at that position;
        # an integer on a frame aligns to the columns. Raises a value error saying the
        # indexer is incompatible with a series if nothing matched.
        # KNOWN ISSUE: the existing docstring says an array is returned, but in some cases a series is returned.
        """
        Parameters
        ----------
        indexer : tuple, slice, scalar
            Indexer used to get the locations that will be set to `ser`.
        ser : pd.Series
            Values to assign to the locations specified by `indexer`.
        multiindex_indexer : bool, optional
            Defaults to False. Should be set to True if `indexer` was from
            a `pd.MultiIndex`, to avoid unnecessary broadcasting.

        Returns
        -------
        `np.array` of `ser` broadcast to the appropriate shape for assignment
        to the locations selected by `indexer`
        """
        if isinstance(indexer, (slice, np.ndarray, list, Index)):
            indexer = (indexer,)

        if isinstance(indexer, tuple):
            # flatten np.ndarray indexers
            if (
                len(indexer) == 2
                and isinstance(indexer[1], np.ndarray)
                and indexer[1].dtype == np.bool_
            ):
                indexer = (indexer[0], np.where(indexer[1])[0])

            def ravel(i):
            # [FUNCTION DOC]
            # HELPER (nested): returns the flattened one-dimensional version of an array, or
            # the argument unchanged if it is not an array.
                return i.ravel() if isinstance(i, np.ndarray) else i

            indexer = tuple(map(ravel, indexer))
            aligners = [not com.is_null_slice(idx) for idx in indexer]
            sum_aligners = sum(aligners)
            single_aligner = sum_aligners == 1
            is_frame = self.ndim == 2
            obj = self.obj

            # are we a single alignable value on a non-primary
            # dim (e.g. panel: 1,2, or frame: 0) ?
            # hence need to align to a single axis dimension
            # rather that find all valid dims

            # frame
            if is_frame:
                single_aligner = single_aligner and aligners[0]

            # we have a frame, with multiple indexers on both axes; and a
            # series, so need to broadcast (see GH5206)
            if all(is_sequence(_) or isinstance(_, slice) for _ in indexer):
                ser_values = ser.reindex(obj.axes[0][indexer[0]])._values

                # single indexer
                if len(indexer) > 1 and not multiindex_indexer:
                    if isinstance(indexer[1], slice):
                        len_indexer = len(obj.axes[1][indexer[1]])
                    else:
                        len_indexer = len(indexer[1])
                    ser_values = (
                        np.tile(ser_values, len_indexer).reshape(len_indexer, -1).T
                    )

                return ser_values

            for i, idx in enumerate(indexer):
                ax = obj.axes[i]

                # multiple aligners (or null slices)
                if is_sequence(idx) or isinstance(idx, slice):
                    if single_aligner and com.is_null_slice(idx):
                        continue
                    new_ix = ax[idx]
                    if not is_list_like_indexer(new_ix):
                        new_ix = Index([new_ix])
                    else:
                        new_ix = Index(new_ix)
                    if not len(new_ix) or ser.index.equals(new_ix):
                        if using_cow:
                            return ser
                        return ser._values.copy()

                    return ser.reindex(new_ix)._values

                # 2 dims
                elif single_aligner:
                    # reindex along index
                    ax = self.obj.axes[1]
                    if ser.index.equals(ax) or not len(ax):
                        return ser._values.copy()
                    return ser.reindex(ax)._values

        elif is_integer(indexer) and self.ndim == 1:
            if is_object_dtype(self.obj.dtype):
                return ser
            ax = self.obj._get_axis(0)

            if ser.index.equals(ax):
                return ser._values.copy()

            return ser.reindex(ax)._values[indexer]

        elif is_integer(indexer):
            ax = self.obj._get_axis(1)

            if ser.index.equals(ax):
                return ser._values.copy()

            return ser.reindex(ax)._values

        raise ValueError("Incompatible indexer with Series")

    def _align_frame(self, indexer, df: DataFrame) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: align a frame to the locations that an indexer selects. Parameters:
        # indexer; df (frame to assign). For a tuple of row and column indexers with
        # sequence or slice elements, builds the selected row and column labels and returns
        # a copy of df if they already equal df's labels, else df reindexed to them. For
        # a single slice or list-like indexer on a frame, selects the matching rows: a copy
        # of df if the index equals, else df reindexed (raising a type error if a
        # multi-index is aligned with a differing number of levels, since join levels
        # would be needed). Raises a value error saying the indexer is incompatible with a
        # frame in other cases.
        is_frame = self.ndim == 2

        if isinstance(indexer, tuple):
            idx, cols = None, None
            sindexers = []
            for i, ix in enumerate(indexer):
                ax = self.obj.axes[i]
                if is_sequence(ix) or isinstance(ix, slice):
                    if isinstance(ix, np.ndarray):
                        ix = ix.reshape(-1)
                    if idx is None:
                        idx = ax[ix]
                    elif cols is None:
                        cols = ax[ix]
                    else:
                        break
                else:
                    sindexers.append(i)

            if idx is not None and cols is not None:
                if df.index.equals(idx) and df.columns.equals(cols):
                    val = df.copy()
                else:
                    val = df.reindex(idx, columns=cols)
                return val

        elif (isinstance(indexer, slice) or is_list_like_indexer(indexer)) and is_frame:
            ax = self.obj.index[indexer]
            if df.index.equals(ax):
                val = df.copy()
            else:
                # we have a multi-index and are trying to align
                # with a particular, level GH3738
                if (
                    isinstance(ax, MultiIndex)
                    and isinstance(df.index, MultiIndex)
                    and ax.nlevels != df.index.nlevels
                ):
                    raise TypeError(
                        "cannot align on a multi-index with out "
                        "specifying the join levels"
                    )

                val = df.reindex(index=ax)
            return val

        raise ValueError("Incompatible indexer with DataFrame")


class _ScalarAccessIndexer(NDFrameIndexerBase):
    # [CLASS DOC]
    # Base class for fast scalar access (the at and iat indexers); subclass of the
    # compiled indexer base. Class-level state: a flag saying whether keys are
    # positions (set by subclasses). Provides get and set of single values by
    # delegating to the owner's single-value get and set methods, after
    # key conversion defined by subclasses.
    """
    Access scalars quickly.
    """

    # sub-classes need to set _takeable
    _takeable: bool

    def _convert_key(self, key):
        # [FUNCTION DOC]
        # Abstract hook: convert a key tuple into the form the owner's value access
        # expects. Parameter key. The base raises the abstract method error.
        raise AbstractMethodError(self)

    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Get a single value. Parameter key: a tuple, or a single scalar (wrapped in a
        # tuple). A list-like non-tuple key raises a value error (invalid call for
        # scalar access). The key is converted and the owner's scalar getter is called with
        # the key elements and the positional flag. Returns the value.
        if not isinstance(key, tuple):
            # we could have a convertible item here (e.g. Timestamp)
            if not is_list_like_indexer(key):
                key = (key,)
            else:
                raise ValueError("Invalid call for scalar access (getting)!")

        key = self._convert_key(key)
        return self.obj._get_value(*key, takeable=self._takeable)

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Set a single value. Parameters: key (tuple, scalar, or a callable evaluated
        # on the owner for each element); value. A non-tuple key is expanded to a full
        # tuple; the key is converted; a wrong length raises a value error (not enough
        # indexers for scalar setting). Calls the owner's scalar setter. Returns nothing.
        if isinstance(key, tuple):
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            # scalar callable may return tuple
            key = com.apply_if_callable(key, self.obj)

        if not isinstance(key, tuple):
            key = _tuplify(self.ndim, key)
        key = list(self._convert_key(key))
        if len(key) != self.ndim:
            raise ValueError("Not enough indexers for scalar access (setting)!")

        self.obj._set_value(*key, value=value, takeable=self._takeable)


@doc(IndexingMixin.at)
class _AtIndexer(_ScalarAccessIndexer):
    # [CLASS DOC]
    # Scalar label-based indexer (the object behind the at accessor). Subclass of the
    # scalar-access base; its docs are copied from the at property. Keys are labels
    # (not positions). Falls back to the general label indexer for frames whose
    # axes have duplicates.
    _takeable = False

    def _convert_key(self, key):
        # [FUNCTION DOC]
        # HELPER: normalize the key for scalar label access. Parameter key (tuple). For
        # a series with a key of more than one element the key is wrapped as a single
        # label (so that tuple labels of a multi-index work); otherwise unchanged.
        # Returns the key.
        # KNOWN ISSUE: the existing docstring says the keys must be the same type as the index to avoid fallback, but no type check is done; the method only wraps tuple labels for series.
        """
        Require they keys to be the same type as the index. (so we don't
        fallback)
        """
        # GH 26989
        # For series, unpacking key needs to result in the label.
        # This is already the case for len(key) == 1; e.g. (1,)
        if self.ndim == 1 and len(key) > 1:
            key = (key,)

        return key

    @property
    def _axes_are_unique(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property for frames (asserted two-dimensional): true if both the row
        # index and the columns are unique.
        # Only relevant for self.ndim == 2
        assert self.ndim == 2
        return self.obj.index.is_unique and self.obj.columns.is_unique

    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Get a single value by label. Parameter key. For frames with duplicate axis
        # labels, the key must be a tuple of scalars (else a value error) and the lookup
        # goes through the label indexer; otherwise defers to the base scalar get.
        if self.ndim == 2 and not self._axes_are_unique:
            # GH#33041 fall back to .loc
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                raise ValueError("Invalid call for scalar access (getting)!")
            return self.obj.loc[key]

        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Set a single value by label. Parameters: key; value. Emits a chained assignment
        # warning (stack level two) if the check is enabled and the owner has few
        # references. For frames with duplicate axis labels, requires a tuple of scalars
        # (else a value error) and sets through the label indexer; otherwise defers to the
        # base scalar set.
        if not CHAINED_WARNING_DISABLED:
            if sys.getrefcount(self.obj) <= REF_COUNT_IDX:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )

        if self.ndim == 2 and not self._axes_are_unique:
            # GH#33041 fall back to .loc
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                raise ValueError("Invalid call for scalar access (setting)!")

            self.obj.loc[key] = value
            return

        return super().__setitem__(key, value)


@doc(IndexingMixin.iat)
class _iAtIndexer(_ScalarAccessIndexer):
    # [CLASS DOC]
    # Scalar position-based indexer (the object behind the iat accessor). Subclass of
    # the scalar-access base; its docs are copied from the iat property. Keys are
    # integer positions.
    _takeable = True

    def _convert_key(self, key):
        # [FUNCTION DOC]
        # HELPER: validate the key for scalar position access. Parameter key (iterable of
        # elements). Raises a value error if any element is not an integer. Returns the key
        # unchanged.
        # KNOWN ISSUE: the existing docstring says the arguments are converted to labels, but they are returned unchanged.
        """
        Require integer args. (and convert to label arguments)
        """
        for i in key:
            if not is_integer(i):
                raise ValueError("iAt based indexing can only have integer indexers")
        return key

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC]
        # Set a single value by position. Parameters: key; value. Emits a chained
        # assignment warning (stack level two) if the check is enabled and the owner has
        # few references, then defers to the base scalar set.
        if not CHAINED_WARNING_DISABLED:
            if sys.getrefcount(self.obj) <= REF_COUNT_IDX:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )

        return super().__setitem__(key, value)


def _tuplify(ndim: int, loc: Hashable) -> tuple[Hashable | slice, ...]:
    # [FUNCTION DOC]
    # HELPER: build a full-dimension tuple from an indexer for the first axis.
    # Parameters: ndim; loc (the first-axis indexer). Returns a tuple of length ndim
    # with loc first and full slices after.
    """
    Given an indexer for the first dimension, create an equivalent tuple
    for indexing over all dimensions.

    Parameters
    ----------
    ndim : int
    loc : object

    Returns
    -------
    tuple
    """
    _tup: list[Hashable | slice]
    _tup = [slice(None, None) for _ in range(ndim)]
    _tup[0] = loc
    return tuple(_tup)


def _tupleize_axis_indexer(ndim: int, axis: AxisInt, key) -> tuple:
    # [FUNCTION DOC]
    # HELPER: express an indexer for one axis as a full tuple. Parameters: ndim;
    # axis (integer axis the key refers to); key. Returns a tuple of length ndim of full
    # slices with the key in the position of the axis.
    """
    If we have an axis, adapt the given key to be axis-independent.
    """
    new_key = [slice(None)] * ndim
    new_key[axis] = key
    return tuple(new_key)


def check_bool_indexer(index: Index, key) -> np.ndarray:
    # [FUNCTION DOC]
    # Validate a boolean indexer against an index and align it if needed.
    # Parameters: index (the Index of the indexed object); key (list-like boolean
    # indexer; the caller guarantees it is a boolean indexer). A series whose index
    # differs is aligned to the index; if some labels are missing an indexing error
    # says the boolean series is unalignable; the aligned result is returned as a
    # plain boolean array unless it has an extension dtype. Object-dtype keys become
    # boolean arrays; non-array-likes (which may contain missing) become boolean
    # extension arrays; then the array indexer checker validates the length (raising
    # an index error for a length mismatch). Returns the checked array.
    """
    Check if key is a valid boolean indexer for an object with such index and
    perform reindexing or conversion if needed.

    This function assumes that is_bool_indexer(key) == True.

    Parameters
    ----------
    index : Index
        Index of the object on which the indexing is done.
    key : list-like
        Boolean indexer to check.

    Returns
    -------
    np.array
        Resulting key.

    Raises
    ------
    IndexError
        If the key does not have the same length as index.
    IndexingError
        If the index of the key is unalignable to index.
    """
    result = key
    if isinstance(key, ABCSeries) and not key.index.equals(index):
        indexer = result.index.get_indexer_for(index)
        if -1 in indexer:
            raise IndexingError(
                "Unalignable boolean Series provided as "
                "indexer (index of the boolean Series and of "
                "the indexed object do not match)."
            )

        result = result.take(indexer)

        # fall through for boolean
        if not isinstance(result.dtype, ExtensionDtype):
            return result.astype(bool)._values

    if is_object_dtype(key):
        # key might be object-dtype bool, check_array_indexer needs bool array
        result = np.asarray(result, dtype=bool)
    elif not is_array_like(result):
        # GH 33924
        # key may contain nan elements, check_array_indexer needs bool array
        result = pd_array(result, dtype=bool)
    return check_array_indexer(index, result)


def convert_missing_indexer(indexer):
    # [FUNCTION DOC]
    # Convert a missing-key marker back to the plain key. Parameter indexer. If it is
    # a mapping with the single key entry, returns the pair of (the key, true); a bare
    # boolean key raises a key error (cannot use a single bool to index into setitem).
    # Otherwise returns (indexer, false).
    """
    Reverse convert a missing indexer, which is a dict
    return the scalar indexer and a boolean indicating if we converted
    """
    if isinstance(indexer, dict):
        # a missing key (but not a tuple indexer)
        indexer = indexer["key"]

        if isinstance(indexer, bool):
            raise KeyError("cannot use a single bool to index into setitem")
        return indexer, True

    return indexer, False


def convert_from_missing_indexer_tuple(indexer: tuple, axes: list[Index]) -> tuple:
    # [FUNCTION DOC]
    # Replace missing-key markers in a tuple indexer by positions. Parameters: indexer
    # (tuple); axes (list of Index). Each marker is replaced by the location of its key
    # on the corresponding axis; other elements are kept. Returns a tuple.
    """
    Create a filtered indexer that doesn't have any missing indexers.
    """

    def get_indexer(_i, _idx):
        # [FUNCTION DOC]
        # HELPER (nested): returns the location of the marker's key on the axis at the
        # given position if the element is a marker mapping, else the element itself.
        return axes[_i].get_loc(_idx["key"]) if isinstance(_idx, dict) else _idx

    return tuple(get_indexer(_i, _idx) for _i, _idx in enumerate(indexer))


def maybe_convert_ix(*args):
    # [FUNCTION DOC]
    # Take the cross product of the arguments for open-mesh indexing if all are
    # arrays, lists, series or indexes (using the numpy open-mesh routine);
    # otherwise returns the arguments tuple unchanged.
    """
    We likely want to take the cross-product.
    """
    for arg in args:
        if not isinstance(arg, (np.ndarray, list, ABCSeries, Index)):
            return args
    return np.ix_(*args)


def is_nested_tuple(tup, labels) -> bool:
    # [FUNCTION DOC]
    # True if the key is a tuple containing a list-like or a slice at its first such
    # position and the labels are a multi-index; false otherwise (including for
    # non-tuples). Parameters: tup; labels.
    """
    Returns
    -------
    bool
    """
    # check for a compatible nested tuple and multiindexes among the axes
    if not isinstance(tup, tuple):
        return False

    for k in tup:
        if is_list_like(k) or isinstance(k, slice):
            return isinstance(labels, MultiIndex)

    return False


def is_label_like(key) -> bool:
    # [FUNCTION DOC]
    # True if the key denotes a single label or row: not a slice, not a list-like
    # indexer and not an ellipsis. Parameter key.
    """
    Returns
    -------
    bool
    """
    # select a label or row
    return (
        not isinstance(key, slice)
        and not is_list_like_indexer(key)
        and key is not Ellipsis
    )


def need_slice(obj: slice) -> bool:
    # [FUNCTION DOC]
    # True if the slice actually restricts anything: a start or stop is given, or
    # the step is given and differs from one. Parameter obj: a slice.
    """
    Returns
    -------
    bool
    """
    return (
        obj.start is not None
        or obj.stop is not None
        or (obj.step is not None and obj.step != 1)
    )


def check_dict_or_set_indexers(key) -> None:
    # [FUNCTION DOC]
    # Raise a type error if the key is, or is a tuple containing, a set or a dict
    # (unsupported as indexers; the message suggests using a list). Parameter key.
    # Returns nothing.
    """
    Check if the indexer is or contains a dict or set, which is no longer allowed.
    """
    if isinstance(key, set) or (
        isinstance(key, tuple) and any(isinstance(x, set) for x in key)
    ):
        raise TypeError(
            "Passing a set as an indexer is not supported. Use a list instead."
        )

    if isinstance(key, dict) or (
        isinstance(key, tuple) and any(isinstance(x, dict) for x in key)
    ):
        raise TypeError(
            "Passing a dict as an indexer is not supported. Use a list instead."
        )
