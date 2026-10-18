# [MODULE DOC]
# Pivoting and cross-tabulation routines.
# Public contents: pivot_table (grouped aggregation arranged as a spreadsheet-style
# table, with optional margins), pivot (reshape without aggregation, requiring unique
# index/column pairs) and crosstab (frequency or aggregated table of factors). All are
# exported to the top-level namespace via the reshape api module; DataFrame methods of
# the same names delegate here.
# Private helpers implement the single-aggregation pivot table, margin (total row and
# column) computation, normalization for crosstab, and name handling.
# Relationships: relies on groupby aggregation, unstack, concat, and index combination
# helpers; crosstab is implemented on top of pivot_table.
# No import-time side effects beyond imports.
from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)

import numpy as np

from pandas._libs import lib
from pandas.util._decorators import set_module

from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import (
    is_list_like,
    is_nested_list_like,
    is_scalar,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

import pandas.core.common as com
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    get_objs_combined_axis,
)
from pandas.core.reshape.concat import concat
from pandas.core.series import Series

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
    )

    from pandas._typing import (
        AggFuncType,
        AggFuncTypeBase,
        AggFuncTypeDict,
        IndexLabel,
        SequenceNotStr,
    )

    from pandas import DataFrame


@set_module("pandas")
def pivot_table(
    data: DataFrame,
    values=None,
    index=None,
    columns=None,
    aggfunc: AggFuncType = "mean",
    fill_value=None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: Hashable = "All",
    observed: bool = True,
    sort: bool = True,
    **kwargs,
) -> DataFrame:
    # [FUNCTION DOC]
    # Builds a spreadsheet-style pivot table: groups the data by the index and columns
    # keys, aggregates the value columns, and unstacks the column keys into columns.
    # Inputs: data (DataFrame); values (column label or list of labels to aggregate;
    # default all columns not used as keys); index and columns (keys to group by for the
    # rows and columns; each may be a column label, a grouper, an array, or a list of
    # those; none means no keys); aggfunc (function, name, list of these, or mapping
    # from column to function(s); default the string for mean); fill_value (replaces
    # missing cells after aggregation; default none); margins (add total row and column
    # named by margins_name); dropna (drop all-missing columns and, for margins, rows
    # with any missing value; also controls whether missing keys form groups, and when
    # false the full product of index and column levels is retained); margins_name
    # (label of the totals, default All); observed (restrict categorical groupers to
    # observed values, default true); sort (sort the keys and the resulting columns,
    # default true); extra keyword arguments are forwarded to the aggregation function.
    # When aggfunc is a list, one table is built per function and they are concatenated
    # along columns under keys that are the function names (or the function itself when
    # it has no name). Otherwise a single table is built by the internal helper.
    # Returns the resulting DataFrame, with metadata propagated from data using the
    # finalize hook with method "pivot_table".
    # KNOWN ISSUE: the docstring gives the default of observed as false, while the
    # signature default is true (the version note in the same entry says true).
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    The levels in the pivot table will be stored in MultiIndex objects
    (hierarchical indexes) on the index and columns of the result DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame object.
    values : list-like or scalar, optional
        Column or columns to aggregate.
    index : column, Grouper, array, or sequence of the previous
        Keys to group by on the pivot table index. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.
    columns : column, Grouper, array, or sequence of the previous
        Keys to group by on the pivot table column. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.
    aggfunc : function, list of functions, dict, default "mean"
        If a list of functions is passed, the resulting pivot table will have
        hierarchical columns whose top level are the function names
        (inferred from the function objects themselves).
        If a dict is passed, the key is column to aggregate and the value is
        function or list of functions. If ``margins=True``, aggfunc will be
        used to calculate the partial aggregates.
    fill_value : scalar, default None
        Value to replace missing values with (in the resulting pivot table,
        after aggregation).
    margins : bool, default False
        If ``margins=True``, special ``All`` columns and rows
        will be added with partial group aggregates across the categories
        on the rows and columns.
    dropna : bool, default True
        Do not include columns whose entries are all NaN. If True,

        * rows with an NA value in any column will be omitted before computing margins,
        * index/column keys containing NA values will be dropped (see ``dropna``
          parameter in :meth:``DataFrame.groupby``).

    margins_name : str, default 'All'
        Name of the row / column that will contain the totals
        when margins is True.
    observed : bool, default False
        This only applies if any of the groupers are Categoricals.
        If True: only show observed values for categorical groupers.
        If False: show all values for categorical groupers.

        .. versionchanged:: 3.0.0

            The default value is now ``True``.

    sort : bool, default True
        Specifies if the result should be sorted.

    **kwargs : dict
        Optional keyword arguments to pass to ``aggfunc``.

        .. versionadded:: 3.0.0

    Returns
    -------
    DataFrame
        An Excel style pivot table.

    See Also
    --------
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.melt: Unpivot a DataFrame from wide to long format,
        optionally leaving identifiers set.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.pivot>` for more examples.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
    ...         "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
    ...         "C": [
    ...             "small",
    ...             "large",
    ...             "large",
    ...             "small",
    ...             "small",
    ...             "large",
    ...             "small",
    ...             "small",
    ...             "large",
    ...         ],
    ...         "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
    ...         "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
    ...     }
    ... )
    >>> df
         A    B      C  D  E
    0  foo  one  small  1  2
    1  foo  one  large  2  4
    2  foo  one  large  2  5
    3  foo  two  small  3  5
    4  foo  two  small  3  6
    5  bar  one  large  4  6
    6  bar  one  small  5  8
    7  bar  two  small  6  9
    8  bar  two  large  7  9

    This first example aggregates values by taking the sum.

    >>> table = pd.pivot_table(
    ...     df, values="D", index=["A", "B"], columns=["C"], aggfunc="sum"
    ... )
    >>> table
    C        large  small
    A   B
    bar one    4.0    5.0
        two    7.0    6.0
    foo one    4.0    1.0
        two    NaN    6.0

    We can also fill missing values using the `fill_value` parameter.

    >>> table = pd.pivot_table(
    ...     df, values="D", index=["A", "B"], columns=["C"], aggfunc="sum", fill_value=0
    ... )
    >>> table
    C        large  small
    A   B
    bar one      4      5
        two      7      6
    foo one      4      1
        two      0      6

    The next example aggregates by taking the mean across multiple columns.

    >>> table = pd.pivot_table(
    ...     df, values=["D", "E"], index=["A", "C"], aggfunc={"D": "mean", "E": "mean"}
    ... )
    >>> table
                    D         E
    A   C
    bar large  5.500000  7.500000
        small  5.500000  8.500000
    foo large  2.000000  4.500000
        small  2.333333  4.333333

    We can also calculate multiple types of aggregations for any given
    value column.

    >>> table = pd.pivot_table(
    ...     df,
    ...     values=["D", "E"],
    ...     index=["A", "C"],
    ...     aggfunc={"D": "mean", "E": ["min", "max", "mean"]},
    ... )
    >>> table
                      D   E
                   mean max      mean  min
    A   C
    bar large  5.500000   9  7.500000    6
        small  5.500000   9  8.500000    8
    foo large  2.000000   5  4.500000    4
        small  2.333333   6  4.333333    2
    """
    index = _convert_by(index)
    columns = _convert_by(columns)

    if isinstance(aggfunc, list):
        pieces: list[DataFrame] = []
        keys = []
        for func in aggfunc:
            _table = __internal_pivot_table(
                data,
                values=values,
                index=index,
                columns=columns,
                fill_value=fill_value,
                aggfunc=func,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
                kwargs=kwargs,
            )
            pieces.append(_table)
            keys.append(getattr(func, "__name__", func))

        table = concat(pieces, keys=keys, axis=1)
        return table.__finalize__(data, method="pivot_table")

    table = __internal_pivot_table(
        data,
        values,
        index,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
        kwargs,
    )
    return table.__finalize__(data, method="pivot_table")


def __internal_pivot_table(
    data: DataFrame,
    values,
    index,
    columns,
    aggfunc: AggFuncTypeBase | AggFuncTypeDict,
    fill_value,
    margins: bool,
    dropna: bool,
    margins_name: Hashable,
    observed: bool,
    sort: bool,
    kwargs,
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: for pivot_table; builds one pivot table for a single (non-list)
    # aggregation. Arguments mirror pivot_table, except index and columns are already
    # normalized to lists and the extra function arguments arrive as one mapping.
    # (The name has two leading underscores and is module-level, so it is not mangled
    # when called from module functions.)
    # Steps: (1) if values were given, normalizes them to a list (remembering whether a
    # single label was passed), raises a key error for a label missing from the data,
    # and narrows the data to the key and value columns; otherwise values are all
    # remaining columns. (2) groups by the index plus column keys and aggregates (with
    # the selected values only when given); with dropna, rows that are entirely missing
    # are dropped. (3) when there is more than one level and index keys exist, unstacks
    # the column-key levels (choosing level positions when names are None or collide
    # with index names). (4) when dropna is false, reindexes onto the full product of
    # the levels. (5) when sort is true, sorts columns. (6) applies fill_value to
    # missing cells, and for a length aggregation with unobserved categories and an
    # integer fill, casts to a 64-bit integer. (7) with margins, drops rows with missing
    # values if dropna, and adds the totals through the margins helper. (8) for a single
    # value label, drops the top column level; if there are no index keys but there are
    # column keys, transposes; with dropna, removes all-missing columns.
    # Returns the table (a DataFrame, or occasionally a Series from the aggregation).
    # KNOWN ISSUE: the call that adds margins passes the dropna flag as the observed
    # argument rather than the observed parameter of this function, so the observed
    # setting is ignored for margin computation.
    """
    Helper of :func:`pandas.pivot_table` for any non-list ``aggfunc``.
    """
    keys = index + columns

    values_passed = values is not None
    if values_passed:
        if is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]

        # GH14938 Make sure value labels are in data
        for i in values:
            if i not in data:
                raise KeyError(i)

        to_filter = []
        for x in keys + values:
            if isinstance(x, Grouper):
                x = x.key
            try:
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        if len(to_filter) < len(data.columns):
            data = data[to_filter]

    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)

    grouped = data.groupby(keys, observed=observed, sort=sort, dropna=dropna)
    if values_passed:
        # GH#57876 and GH#61292
        # mypy is not aware `grouped[values]` will always be a DataFrameGroupBy
        grouped = grouped[values]  # type: ignore[assignment]

    agged = grouped.agg(aggfunc, **kwargs)

    if dropna and isinstance(agged, ABCDataFrame) and len(agged.columns):
        agged = agged.dropna(how="all")

    table = agged

    # GH17038, this check should only happen if index is defined (not None)
    if table.index.nlevels > 1 and index:
        # Related GH #17123
        # If index_names are integers, determine whether the integers refer
        # to the level position or name.
        index_names = agged.index.names[: len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = agged.unstack(to_unstack, fill_value=fill_value)

    if not dropna:
        if isinstance(table.index, MultiIndex):
            m = MultiIndex.from_product(table.index.levels, names=table.index.names)
            table = table.reindex(m, axis=0, fill_value=fill_value)

        if isinstance(table.columns, MultiIndex):
            m = MultiIndex.from_product(table.columns.levels, names=table.columns.names)
            table = table.reindex(m, axis=1, fill_value=fill_value)

    if sort is True and isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)

    if fill_value is not None:
        table = table.fillna(fill_value)
        if aggfunc is len and not observed and lib.is_integer(fill_value):
            # TODO: can we avoid this?  this used to be handled by
            #  downcast="infer" in fillna
            table = table.astype(np.int64)

    if margins:
        if dropna:
            data = data[data.notna().all(axis=1)]
        table = _add_margins(
            table,
            data,
            values,
            rows=index,
            cols=columns,
            aggfunc=aggfunc,
            kwargs=kwargs,
            observed=dropna,
            margins_name=margins_name,
            fill_value=fill_value,
            dropna=dropna,
        )

    # discard the top level
    if values_passed and not values_multi and table.columns.nlevels > 1:
        table.columns = table.columns.droplevel(0)
    if len(index) == 0 and len(columns) > 0:
        table = table.T

    # GH 15193 Make sure empty columns are removed if dropna=True
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how="all", axis=1)

    return table


def _add_margins(
    table: DataFrame | Series,
    data: DataFrame,
    values,
    rows,
    cols,
    aggfunc,
    kwargs,
    observed: bool,
    margins_name: Hashable = "All",
    fill_value=None,
    dropna: bool = True,
):
    # [FUNCTION DOC]
    # HELPER: for the internal pivot table; appends the margin (grand total) row and
    # column to an already aggregated table.
    # Inputs: table (DataFrame or Series result), data (original, possibly NA-filtered
    # data), values (list of value columns, possibly empty), rows and cols (lists of
    # row and column keys), aggfunc and kwargs (aggregation and its keyword
    # arguments), observed (passed to the groupby of the margins), margins_name (must be
    # a string, default All), fill_value (for missing margin cells), dropna.
    # Raises a value error if margins_name is not a string, or if the margin name
    # already appears in the table's index levels or in the column levels beyond the
    # first.
    # Cases: no values and a Series table: appends one entry holding the grand margin;
    # with values: the marginal-results helper builds the interleaved margins (it may
    # return the finished table directly); with no values and a DataFrame: the variant
    # helper without values is used. Then the row margin is aligned to the result
    # columns and filled from the grand margins, converted to a one-row frame keyed by
    # the margin label (a tuple padded with empty strings for several row keys), cast
    # back toward each column's original non-extension dtype, and concatenated below
    # the result with the original index names restored.
    # Returns the table with margins.
    if not isinstance(margins_name, str):
        raise ValueError("margins_name argument must be a string")

    msg = f'Conflicting name "{margins_name}" in margins'
    for level in table.index.names:
        if margins_name in table.index.get_level_values(level):
            raise ValueError(msg)

    grand_margin = _compute_grand_margin(data, values, aggfunc, kwargs, margins_name)

    if table.ndim == 2:
        # i.e. DataFrame
        for level in table.columns.names[1:]:
            if margins_name in table.columns.get_level_values(level):
                raise ValueError(msg)

    key: str | tuple[str, ...]
    if len(rows) > 1:
        key = (margins_name,) + ("",) * (len(rows) - 1)
    else:
        key = margins_name

    if not values and isinstance(table, ABCSeries):
        # If there are no values and the table is a series, then there is only
        # one column in the data. Compute grand margin and return it.
        return table._append_internal(
            table._constructor({key: grand_margin[margins_name]})
        )

    elif values:
        marginal_result_set = _generate_marginal_results(
            table,
            data,
            values,
            rows,
            cols,
            aggfunc,
            kwargs,
            observed,
            margins_name,
            dropna,
        )
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set
    else:
        # no values, and table is a DataFrame
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(
            table, data, rows, cols, aggfunc, kwargs, observed, margins_name, dropna
        )
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set

    row_margin = row_margin.reindex(result.columns, fill_value=fill_value)
    # populate grand margin
    for k in margin_keys:
        if isinstance(k, str):
            row_margin[k] = grand_margin[k]
        else:
            row_margin[k] = grand_margin[k[0]]

    from pandas import DataFrame

    margin_dummy = DataFrame(row_margin, columns=Index([key])).T

    row_names = result.index.names
    # check the result column and leave floats

    for dtype in set(result.dtypes):
        if isinstance(dtype, ExtensionDtype):
            # Can hold NA already
            continue

        cols = result.select_dtypes([dtype]).columns
        margin_dummy[cols] = margin_dummy[cols].apply(
            maybe_downcast_to_dtype, args=(dtype,)
        )
    result = concat([result, margin_dummy])
    result.index.names = row_names

    return result


def _compute_grand_margin(
    data: DataFrame, values, aggfunc, kwargs, margins_name: Hashable = "All"
):
    # [FUNCTION DOC]
    # HELPER: for the margins step; computes grand totals by aggregating whole columns.
    # Inputs: data, values (list of value columns, possibly empty), aggfunc (name,
    # callable, or mapping from column to name or callable), kwargs (forwarded),
    # margins_name (key used when there are no values).
    # With values: returns a mapping from each value column to its aggregate over all
    # rows; columns for which the aggregation raises a type error are silently left out.
    # Without values: returns a mapping with the single margins_name key whose value is
    # the aggregation applied to the data's index.
    if values:
        grand_margin = {}
        for k, v in data[values].items():
            try:
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)(**kwargs)
                elif isinstance(aggfunc, dict):
                    if isinstance(aggfunc[k], str):
                        grand_margin[k] = getattr(v, aggfunc[k])(**kwargs)
                    else:
                        grand_margin[k] = aggfunc[k](v, **kwargs)
                else:
                    grand_margin[k] = aggfunc(v, **kwargs)
            except TypeError:
                pass
        return grand_margin
    else:
        return {margins_name: aggfunc(data.index, **kwargs)}


def _generate_marginal_results(
    table,
    data: DataFrame,
    values,
    rows,
    cols,
    aggfunc,
    kwargs,
    observed: bool,
    margins_name: Hashable = "All",
    dropna: bool = True,
):
    # [FUNCTION DOC]
    # HELPER: for the margins step when value columns exist; computes the per-group
    # partial margins and interleaves them into the table.
    # Inputs: table (aggregated table), data, values (list), rows and cols (key lists),
    # aggfunc, kwargs, observed, margins_name, dropna.
    # With column keys: aggregates the data by the row keys (or, if none, by the first
    # column key) to get margins, and for each top-level group of columns (or rows when
    # no row keys) appends a margin column or row labeled with the margin name padded
    # with empty strings for extra levels; the pieces are concatenated. If no pieces
    # exist the table is returned unchanged; if there are no row keys the concatenated
    # result is returned directly (a lone result, not a tuple). Without column keys the
    # table is its own result and the margin keys are its columns.
    # The row margin is the aggregation by column keys, stacked and with levels
    # reordered so the value level comes first (by level names, to support numeric
    # names); with no column keys it is a missing-value series over the result columns.
    # Returns either the finished table, or a tuple of the result, the list of margin
    # keys, and the row margin.
    margin_keys: list | Index
    if len(cols) > 0:
        # need to "interleave" the margins
        table_pieces = []
        margin_keys = []

        def _all_key(key):
            # [FUNCTION DOC]
            # HELPER: nested in the marginal-results helper; builds the label of a
            # margin column or row for one group: the group key, then the margin name,
            # then empty strings padding out the remaining column levels. Returns a
            # tuple.
            return (key, margins_name) + ("",) * (len(cols) - 1)

        if len(rows) > 0:
            margin = (
                data[rows + values]
                .groupby(rows, observed=observed, dropna=dropna)
                .agg(aggfunc, **kwargs)
            )
            cat_axis = 1

            for key, piece in table.T.groupby(level=0, observed=observed):
                piece = piece.T
                all_key = _all_key(key)

                piece[all_key] = margin[key]

                table_pieces.append(piece)
                margin_keys.append(all_key)
        else:
            margin = (
                data[cols[:1] + values]
                .groupby(cols[:1], observed=observed, dropna=dropna)
                .agg(aggfunc, **kwargs)
                .T
            )

            cat_axis = 0
            for key, piece in table.groupby(level=0, observed=observed):
                if len(cols) > 1:
                    all_key = _all_key(key)
                else:
                    all_key = margins_name
                table_pieces.append(piece)
                transformed_piece = margin[key].to_frame().T
                if isinstance(piece.index, MultiIndex):
                    # We are adding an empty level
                    transformed_piece.index = MultiIndex.from_tuples(
                        [all_key],
                        names=[*piece.index.names, None],
                    )
                else:
                    transformed_piece.index = Index([all_key], name=piece.index.name)

                # append piece for margin into table_piece
                table_pieces.append(transformed_piece)
                margin_keys.append(all_key)

        if not table_pieces:
            # GH 49240
            return table
        else:
            result = concat(table_pieces, axis=cat_axis)

        if len(rows) == 0:
            return result
    else:
        result = table
        margin_keys = table.columns

    if len(cols) > 0:
        row_margin = (
            data[cols + values]
            .groupby(cols, observed=observed, dropna=dropna)
            .agg(aggfunc, **kwargs)
        )
        row_margin = row_margin.stack()

        # GH#26568. Use names instead of indices in case of numeric names
        new_order_indices = itertools.chain([len(cols)], range(len(cols)))
        new_order_names = [row_margin.index.names[i] for i in new_order_indices]
        row_margin.index = row_margin.index.reorder_levels(new_order_names)
    else:
        row_margin = data._constructor_sliced(np.nan, index=result.columns)

    return result, margin_keys, row_margin


def _generate_marginal_results_without_values(
    table: DataFrame,
    data: DataFrame,
    rows,
    cols,
    aggfunc,
    kwargs,
    observed: bool,
    margins_name: Hashable = "All",
    dropna: bool = True,
):
    # [FUNCTION DOC]
    # HELPER: for the margins step when no value columns were given; computes margins
    # for a table of counts of the key columns.
    # Inputs: table (DataFrame), data, rows and cols (key lists), aggfunc, kwargs,
    # observed, margins_name, dropna.
    # With column keys: the aggregation function is applied per row-key group (or per
    # first index level when no row keys) and stored as an added margin column in the
    # table itself (mutating the passed table); with no row keys the table is returned
    # directly. Without column keys the table is its own result.
    # Row margin: the aggregation applied per column-key group, or a missing-value
    # series over the result columns when there are no column keys.
    # Returns either the table alone (no row keys) or a tuple of the result, the list
    # of margin keys, and the row margin.
    margin_keys: list | Index
    if len(cols) > 0:
        # need to "interleave" the margins
        margin_keys = []

        def _all_key():
            # [FUNCTION DOC]
            # HELPER: nested in the without-values marginal helper; returns the margin
            # label: the bare margin name for one column key, else a tuple of the name
            # followed by empty strings padding the extra levels.
            if len(cols) == 1:
                return margins_name
            return (margins_name,) + ("",) * (len(cols) - 1)

        if len(rows) > 0:
            margin = data.groupby(rows, observed=observed, dropna=dropna)[rows].apply(
                aggfunc, **kwargs
            )
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)

        else:
            margin = data.groupby(level=0, observed=observed, dropna=dropna).apply(
                aggfunc, **kwargs
            )
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
            return result
    else:
        result = table
        margin_keys = table.columns

    if len(cols):
        row_margin = data.groupby(cols, observed=observed, dropna=dropna)[cols].apply(
            aggfunc, **kwargs
        )
    else:
        row_margin = Series(np.nan, index=result.columns)

    return result, margin_keys, row_margin


def _convert_by(by):
    # [FUNCTION DOC]
    # HELPER: for pivot_table; normalizes an index or columns argument to a list.
    # None becomes an empty list; a scalar, array, index, series, grouper or callable is
    # wrapped in a one-element list; any other iterable is converted to a list.
    if by is None:
        by = []
    elif (
        is_scalar(by)
        or isinstance(by, (np.ndarray, Index, ABCSeries, Grouper))
        or callable(by)
    ):
        by = [by]
    else:
        by = list(by)
    return by


@set_module("pandas")
def pivot(
    data: DataFrame,
    *,
    columns: IndexLabel,
    index: IndexLabel | lib.NoDefault = lib.no_default,
    values: IndexLabel | lib.NoDefault = lib.no_default,
) -> DataFrame:
    # [FUNCTION DOC]
    # Reshapes data without aggregation: unique values of the columns key(s) become the
    # new columns and values of the index key(s) become the new row labels.
    # Inputs: data (DataFrame); columns (required keyword; label or list of labels);
    # index (label or list; if not given the existing index, possibly hierarchical, is
    # used); values (label or list; if not given, all remaining columns are used and the
    # result has hierarchical columns). Tuples count as a single column label.
    # Algorithm: unnamed index levels are temporarily given a placeholder name so they
    # do not collide with the none-named level that unstacking creates; then, without
    # values, the key columns are set as the index (appended to the existing index when
    # no index key was given); with values, a hierarchical index is built from the index
    # and column key arrays and the selected values are wrapped in a frame (list-like
    # values) or Series. The column keys are then unstacked and the placeholder names
    # are reset to none. Input is not mutated (a shallow copy is made when renaming).
    # Raises a value error (from unstacking) when an index and columns combination is
    # duplicated.
    # Returns the reshaped DataFrame.
    """
    Return reshaped DataFrame organized by given index / column values.

    Reshape data (produce a "pivot" table) based on column values. Uses
    unique values from specified `index` / `columns` to form axes of the
    resulting DataFrame. This function does not support data
    aggregation, multiple values will result in a MultiIndex in the
    columns. See the :ref:`User Guide <reshaping>` for more on reshaping.

    Parameters
    ----------
    data : DataFrame
        Input pandas DataFrame object.
    columns : Hashable or a sequence of the previous
        Column to use to make new frame's columns.
    index : Hashable or a sequence of the previous, optional
        Column to use to make new frame's index. If not given, uses existing index.
    values : Hashable or a sequence of the previous, optional
        Column(s) to use for populating new frame's values. If not
        specified, all remaining columns will be used and the result will
        have hierarchically indexed columns.

    Returns
    -------
    DataFrame
        Returns reshaped DataFrame.

    Raises
    ------
    ValueError:
        When there are any `index`, `columns` combinations with multiple
        values. `DataFrame.pivot_table` when you need to aggregate.

    See Also
    --------
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Notes
    -----
    For finer-tuned control, see hierarchical indexing documentation along
    with the related stack/unstack methods.

    Reference :ref:`the user guide <reshaping.pivot>` for more examples.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "foo": ["one", "one", "one", "two", "two", "two"],
    ...         "bar": ["A", "B", "C", "A", "B", "C"],
    ...         "baz": [1, 2, 3, 4, 5, 6],
    ...         "zoo": ["x", "y", "z", "q", "w", "t"],
    ...     }
    ... )
    >>> df
        foo   bar  baz  zoo
    0   one   A    1    x
    1   one   B    2    y
    2   one   C    3    z
    3   two   A    4    q
    4   two   B    5    w
    5   two   C    6    t

    >>> df.pivot(index="foo", columns="bar", values="baz")
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index="foo", columns="bar")["baz"]
    bar  A   B   C
    foo
    one  1   2   3
    two  4   5   6

    >>> df.pivot(index="foo", columns="bar", values=["baz", "zoo"])
          baz       zoo
    bar   A  B  C   A  B  C
    foo
    one   1  2  3   x  y  z
    two   4  5  6   q  w  t

    You could also assign a list of column names or a list of index names.

    >>> df = pd.DataFrame(
    ...     {
    ...         "lev1": [1, 1, 1, 2, 2, 2],
    ...         "lev2": [1, 1, 2, 1, 1, 2],
    ...         "lev3": [1, 2, 1, 2, 1, 2],
    ...         "lev4": [1, 2, 3, 4, 5, 6],
    ...         "values": [0, 1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> df
        lev1 lev2 lev3 lev4 values
    0   1    1    1    1    0
    1   1    1    2    2    1
    2   1    2    1    3    2
    3   2    1    2    4    3
    4   2    1    1    5    4
    5   2    2    2    6    5

    >>> df.pivot(index="lev1", columns=["lev2", "lev3"], values="values")
    lev2    1         2
    lev3    1    2    1    2
    lev1
    1     0.0  1.0  2.0  NaN
    2     4.0  3.0  NaN  5.0

    >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")
          lev3    1    2
    lev1  lev2
       1     1  0.0  1.0
             2  2.0  NaN
       2     1  4.0  3.0
             2  NaN  5.0

    A ValueError is raised if there are any duplicates.

    >>> df = pd.DataFrame(
    ...     {
    ...         "foo": ["one", "one", "two", "two"],
    ...         "bar": ["A", "A", "B", "C"],
    ...         "baz": [1, 2, 3, 4],
    ...     }
    ... )
    >>> df
       foo bar  baz
    0  one   A    1
    1  one   A    2
    2  two   B    3
    3  two   C    4

    Notice that the first two rows are the same for our `index`
    and `columns` arguments.

    >>> df.pivot(index="foo", columns="bar", values="baz")
    Traceback (most recent call last):
       ...
    ValueError: Index contains duplicate entries, cannot reshape
    """
    columns_listlike = com.convert_to_list_like(columns)

    # If columns is None we will create a MultiIndex level with None as name
    # which might cause duplicated names because None is the default for
    # level names
    if any(name is None for name in data.index.names):
        data = data.copy(deep=False)
        data.index.names = [
            name if name is not None else lib.no_default for name in data.index.names
        ]

    indexed: DataFrame | Series
    if values is lib.no_default:
        if index is not lib.no_default:
            cols = com.convert_to_list_like(index)
        else:
            cols = []

        append = index is lib.no_default
        # error: Unsupported operand types for + ("List[Any]" and "ExtensionArray")
        # error: Unsupported left operand type for + ("ExtensionArray")
        indexed = data.set_index(
            cols + columns_listlike,  # type: ignore[operator]
            append=append,
        )
    else:
        index_list: list[Index] | list[Series]
        if index is lib.no_default:
            if isinstance(data.index, MultiIndex):
                # GH 23955
                index_list = [
                    data.index.get_level_values(i) for i in range(data.index.nlevels)
                ]
            else:
                index_list = [
                    data._constructor_sliced(data.index, name=data.index.name)
                ]
        else:
            index_list = [data[idx] for idx in com.convert_to_list_like(index)]

        data_columns = [data[col] for col in columns_listlike]
        index_list.extend(data_columns)
        multiindex = MultiIndex.from_arrays(index_list)

        if is_list_like(values) and not isinstance(values, tuple):
            # Exclude tuple because it is seen as a single column name
            indexed = data._constructor(
                data[values]._values,
                index=multiindex,
                columns=cast("SequenceNotStr", values),
            )
        else:
            indexed = data._constructor_sliced(data[values]._values, index=multiindex)
    # error: Argument 1 to "unstack" of "DataFrame" has incompatible type "Union
    # [List[Any], ExtensionArray, ndarray[Any, Any], Index, Series]"; expected
    # "Hashable"
    # unstack with a MultiIndex returns a DataFrame
    result = cast("DataFrame", indexed.unstack(columns_listlike))  # type: ignore[arg-type]
    result.index.names = [
        name if name is not lib.no_default else None for name in result.index.names
    ]

    return result


@set_module("pandas")
def crosstab(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins: bool = False,
    margins_name: Hashable = "All",
    dropna: bool = True,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = False,
) -> DataFrame:
    # [FUNCTION DOC]
    # Computes a cross tabulation of factors: counts by default, or an aggregation of
    # values.
    # Inputs: index and columns (array-like, Series, or list of these, defining the row
    # and column factors); values (array to aggregate; requires aggfunc); rownames and
    # colnames (optional names; the length must match the number of arrays, else an
    # assertion error; by default Series names or generated names like row_0 and col_0
    # are used); aggfunc (aggregation; requires values); margins and margins_name
    # (totals); dropna (drop all-missing columns; also used as the observed setting, so
    # categories without data are kept when false); normalize (false, or true/"all",
    # "index"/0, "columns"/1, to divide counts into proportions).
    # Raises a value error when only one of values and aggfunc is given.
    # Algorithm: when Series inputs are present, the common index is the intersection of
    # their indexes. Names that are duplicated among rows and columns are mapped to
    # temporary unique names. A frame of the factors plus a dummy column (zeros for
    # counting, or the values) is built and passed to pivot_table with a length
    # aggregation and zero fill for counting. Then optional normalization is applied and
    # the original names are restored on the axes.
    # Returns the cross-tabulation DataFrame.
    """
    Compute a simple cross tabulation of two (or more) factors.

    By default, computes a frequency table of the factors unless an
    array of values and an aggregation function are passed.

    Parameters
    ----------
    index : array-like, Series, or list of arrays/Series
        Values to group by in the rows.
    columns : array-like, Series, or list of arrays/Series
        Values to group by in the columns.
    values : array-like, optional
        Array of values to aggregate according to the factors.
        Requires `aggfunc` be specified.
    rownames : sequence, default None
        If passed, must match number of row arrays passed.
    colnames : sequence, default None
        If passed, must match number of column arrays passed.
    aggfunc : function, optional
        If specified, requires `values` be specified as well.
    margins : bool, default False
        Add row/column margins (subtotals).
    margins_name : str, default 'All'
        Name of the row/column that will contain the totals
        when margins is True.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    normalize : bool, {'all', 'index', 'columns'}, or {0,1}, default False
        Normalize by dividing all values by the sum of values.

        - If passed 'all' or `True`, will normalize over all values.
        - If passed 'index' will normalize over each row.
        - If passed 'columns' will normalize over each column.
        - If margins is `True`, will also normalize margin values.

    Returns
    -------
    DataFrame
        Cross tabulation of the data.

    See Also
    --------
    DataFrame.pivot : Reshape data based on column values.
    pivot_table : Create a pivot table as a DataFrame.

    Notes
    -----
    Any Series passed will have their name attributes used unless row or column
    names for the cross-tabulation are specified.

    Any input passed containing Categorical data will have **all** of its
    categories included in the cross-tabulation, even if the actual data does
    not contain any instances of a particular category.

    In the event that there aren't overlapping indexes an empty DataFrame will
    be returned.

    Reference :ref:`the user guide <reshaping.crosstabulations>` for more examples.

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         "foo",
    ...         "foo",
    ...         "foo",
    ...         "foo",
    ...         "bar",
    ...         "bar",
    ...         "bar",
    ...         "bar",
    ...         "foo",
    ...         "foo",
    ...         "foo",
    ...     ],
    ...     dtype=object,
    ... )
    >>> b = np.array(
    ...     [
    ...         "one",
    ...         "one",
    ...         "one",
    ...         "two",
    ...         "one",
    ...         "one",
    ...         "one",
    ...         "two",
    ...         "two",
    ...         "two",
    ...         "one",
    ...     ],
    ...     dtype=object,
    ... )
    >>> c = np.array(
    ...     [
    ...         "dull",
    ...         "dull",
    ...         "shiny",
    ...         "dull",
    ...         "dull",
    ...         "shiny",
    ...         "shiny",
    ...         "dull",
    ...         "shiny",
    ...         "shiny",
    ...         "shiny",
    ...     ],
    ...     dtype=object,
    ... )
    >>> pd.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])
    b   one        two
    c   dull shiny dull shiny
    a
    bar    1     2    1     0
    foo    2     2    1     2

    Here 'c' and 'f' are not represented in the data and will not be
    shown in the output because dropna is True by default. Set
    dropna=False to preserve categories with no data.

    >>> foo = pd.Categorical(["a", "b"], categories=["a", "b", "c"])
    >>> bar = pd.Categorical(["d", "e"], categories=["d", "e", "f"])
    >>> pd.crosstab(foo, bar)
    col_0  d  e
    row_0
    a      1  0
    b      0  1
    >>> pd.crosstab(foo, bar, dropna=False)
    col_0  d  e  f
    row_0
    a      1  0  0
    b      0  1  0
    c      0  0  0
    """
    if values is None and aggfunc is not None:
        raise ValueError("aggfunc cannot be used without values.")

    if values is not None and aggfunc is None:
        raise ValueError("values cannot be used without an aggfunc.")

    if not is_nested_list_like(index):
        index = [index]
    if not is_nested_list_like(columns):
        columns = [columns]

    common_idx = None
    pass_objs = [x for x in index + columns if isinstance(x, (ABCSeries, ABCDataFrame))]
    if pass_objs:
        common_idx = get_objs_combined_axis(pass_objs, intersect=True, sort=False)

    rownames = _get_names(index, rownames, prefix="row")
    colnames = _get_names(columns, colnames, prefix="col")

    # duplicate names mapped to unique names for pivot op
    (
        rownames_mapper,
        unique_rownames,
        colnames_mapper,
        unique_colnames,
    ) = _build_names_mapper(rownames, colnames)

    from pandas import DataFrame

    data = {
        **dict(zip(unique_rownames, index, strict=True)),
        **dict(zip(unique_colnames, columns, strict=True)),
    }
    df = DataFrame(data, index=common_idx)

    if values is None:
        df["__dummy__"] = 0
        kwargs = {"aggfunc": len, "fill_value": 0}
    else:
        df["__dummy__"] = values
        kwargs = {"aggfunc": aggfunc}

    # error: Argument 7 to "pivot_table" of "DataFrame" has incompatible type
    # "**Dict[str, object]"; expected "Union[...]"
    table = df.pivot_table(
        "__dummy__",
        index=unique_rownames,
        columns=unique_colnames,
        margins=margins,
        margins_name=margins_name,
        dropna=dropna,
        observed=dropna,
        **kwargs,  # type: ignore[arg-type]
    )

    # Post-process
    if normalize is not False:
        table = _normalize(
            table, normalize=normalize, margins=margins, margins_name=margins_name
        )

    table = table.rename_axis(index=rownames_mapper, axis=0)
    table = table.rename_axis(columns=colnames_mapper, axis=1)

    return table


def _normalize(
    table: DataFrame, normalize, margins: bool, margins_name: Hashable = "All"
) -> DataFrame:
    # [FUNCTION DOC]
    # HELPER: for crosstab; normalizes a cross-tabulation table into proportions.
    # Inputs: table, normalize (true, "all", "index", "columns", or the integers 0 for
    # index and 1 for columns), margins (whether the table includes margin row and
    # column), margins_name (label of the margins).
    # Integers are translated to names; anything unknown raises a value error. Without
    # margins: divides by the grand total ("all" or true), by the column sums
    # ("columns"), or by the row sums ("index"), using three small anonymous division
    # functions held in a lookup table, then fills missing results with zero. With
    # margins: separates the margin column and row (raising a value error if the margin
    # name is not at the last position), normalizes the core table recursively without
    # margins, normalizes the margins by their own sums, reattaches them (the overall
    # corner is set to one for the "all" case), fills missing with zero and restores the
    # original index and columns. A margins value other than a boolean raises a value
    # error. Returns the normalized table.
    if not isinstance(normalize, (bool, str)):
        axis_subs = {0: "index", 1: "columns"}
        try:
            normalize = axis_subs[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

    if margins is False:
        # Actual Normalizations
        normalizers: dict[bool | str, Callable] = {
            "all": lambda x: x / x.sum(axis=1).sum(axis=0),
            "columns": lambda x: x / x.sum(),
            "index": lambda x: x.div(x.sum(axis=1), axis=0),
        }

        normalizers[True] = normalizers["all"]

        try:
            f = normalizers[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

        table = f(table)
        table = table.fillna(0)

    elif margins is True:
        # keep index and column of pivoted table
        table_index = table.index
        table_columns = table.columns
        last_ind_or_col = table.iloc[-1, :].name

        # check if margin name is not in (for MI cases) and not equal to last
        # index/column and save the column and index margin
        if (margins_name not in last_ind_or_col) & (margins_name != last_ind_or_col):
            raise ValueError(f"{margins_name} not in pivoted DataFrame")
        column_margin = table.iloc[:-1, -1]
        index_margin = table.iloc[-1, :-1]

        # keep the core table
        table = table.iloc[:-1, :-1]

        # Normalize core
        table = _normalize(table, normalize=normalize, margins=False)

        # Fix Margins
        if normalize == "columns":
            column_margin = column_margin / column_margin.sum()
            table = concat([table, column_margin], axis=1)
            table = table.fillna(0)
            table.columns = table_columns

        elif normalize == "index":
            index_margin = index_margin / index_margin.sum()
            table = table._append_internal(index_margin, ignore_index=True)
            table = table.fillna(0)
            table.index = table_index

        elif normalize == "all" or normalize is True:
            column_margin = column_margin / column_margin.sum()
            index_margin = index_margin / index_margin.sum()
            index_margin.loc[margins_name] = 1
            table = concat([table, column_margin], axis=1)
            table = table._append_internal(index_margin, ignore_index=True)

            table = table.fillna(0)
            table.index = table_index
            table.columns = table_columns

        else:
            raise ValueError("Not a valid normalize argument")

    else:
        raise ValueError("Not a valid margins argument")

    return table


def _get_names(arrs, names, prefix: str = "row") -> list:
    # [FUNCTION DOC]
    # HELPER: for crosstab; determines the names of the factor arrays.
    # Inputs: arrs (list of arrays or Series), names (optional user names), prefix
    # (word used for generated names).
    # If names is none: each Series with a name keeps it, others get the prefix, an
    # underscore and their position. Otherwise names must have the same length as arrs
    # (assertion error if not) and is converted to a list. Returns the list of names.
    if names is None:
        names = []
        for i, arr in enumerate(arrs):
            if isinstance(arr, ABCSeries) and arr.name is not None:
                names.append(arr.name)
            else:
                names.append(f"{prefix}_{i}")
    else:
        if len(names) != len(arrs):
            raise AssertionError("arrays and names must have the same length")
        if not isinstance(names, list):
            names = list(names)

    return names


def _build_names_mapper(
    rownames: list[str], colnames: list[str]
) -> tuple[dict[str, str], list[str], dict[str, str], list[str]]:
    # [FUNCTION DOC]
    # HELPER: for crosstab; makes row and column names unique so the intermediate pivot
    # does not clash, and records how to restore them. Any name that appears in either
    # list (a name is considered duplicated if it is present among the row names or
    # the column names) is replaced by a generated positional name; returns the row and
    # column mappers (generated to original) and the unique name lists.
    # KNOWN ISSUE: the replacement test checks membership in the union of row and column
    # names, which is true for every name, so all names are replaced by generated ones
    # rather than only the duplicated ones that the docstring describes.
    """
    Given the names of a DataFrame's rows and columns, returns a set of unique row
    and column names and mappers that convert to original names.

    A row or column name is replaced if it is duplicate among the rows of the inputs,
    among the columns of the inputs or between the rows and the columns.

    Parameters
    ----------
    rownames: list[str]
    colnames: list[str]

    Returns
    -------
    Tuple(Dict[str, str], List[str], Dict[str, str], List[str])

    rownames_mapper: dict[str, str]
        a dictionary with new row names as keys and original rownames as values
    unique_rownames: list[str]
        a list of rownames with duplicate names replaced by dummy names
    colnames_mapper: dict[str, str]
        a dictionary with new column names as keys and original column names as values
    unique_colnames: list[str]
        a list of column names with duplicate names replaced by dummy names

    """
    dup_names = set(rownames) | set(colnames)

    rownames_mapper = {
        f"row_{i}": name for i, name in enumerate(rownames) if name in dup_names
    }
    unique_rownames = [
        f"row_{i}" if name in dup_names else name for i, name in enumerate(rownames)
    ]

    colnames_mapper = {
        f"col_{i}": name for i, name in enumerate(colnames) if name in dup_names
    }
    unique_colnames = [
        f"col_{i}" if name in dup_names else name for i, name in enumerate(colnames)
    ]

    return rownames_mapper, unique_rownames, colnames_mapper, unique_colnames
