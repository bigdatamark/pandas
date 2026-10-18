# [MODULE DOC]
# Implementation of the DataFrame conversion to a dictionary-like structure, in all supported orientations.
# Public contents: the conversion function (with typed overloads for static checkers) and a private-style helper generator that builds row data for the split orientation.
# Relationships: called by the frame method of the same purpose; uses the generic mapping standardization utility and the helper that converts numpy scalars into native Python objects. Emits a user warning for non-unique columns. No import-time side effects.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import maybe_box_native
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    ExtensionDtype,
)

from pandas.core import common as com

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas._typing import MutableMappingT

    from pandas import DataFrame


def create_data_for_split(
    df: DataFrame, are_all_object_dtype_cols: bool, object_dtype_indices: list[int]
) -> Generator[list]:
    # [FUNCTION DOC]
    # HELPER: generator used by the split orientation of the dictionary conversion to produce the data rows.
    # Parameters: df is the frame; are_all_object_dtype_cols says whether every column is object or extension typed; object_dtype_indices is the list of positional column indexes that are object or extension typed.
    # Yields one list per row (excluding the index). When all columns are object/extension typed every value is converted to a native Python object; otherwise only values in the listed positions are converted, after the row is turned into a list, for performance. No side effects.
    """
    Simple helper method to create data for to ``to_dict(orient="split")``
    to create the main output data
    """
    if are_all_object_dtype_cols:
        for tup in df.itertuples(index=False, name=None):
            yield list(map(maybe_box_native, tup))
    else:
        for tup in df.itertuples(index=False, name=None):
            data = list(tup)
            if object_dtype_indices:
                # If we have object_dtype_cols, apply maybe_box_naive after
                # for perf
                for i in object_dtype_indices:
                    data[i] = maybe_box_native(data[i])
            yield data


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> MutableMappingT:
    # [FUNCTION DOC]
    # Typed overload stub for the case with a mapping type or instance given for into and any orientation except records: returns a mapping of that type. The behavior is defined in the implementation below.
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> list[MutableMappingT]:
    # [FUNCTION DOC]
    # Typed overload stub for the records orientation with a mapping type or instance given for into: returns a list of mappings of that type. The behavior is defined in the implementation below.
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> dict:
    # [FUNCTION DOC]
    # Typed overload stub for the default mapping type with any orientation except records: returns a plain dict. The behavior is defined in the implementation below.
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> list[dict]:
    # [FUNCTION DOC]
    # Typed overload stub for the records orientation with the default mapping type: returns a list of plain dicts. The behavior is defined in the implementation below.
    ...


# error: Incompatible default for argument "into" (default has type "type[dict
# [Any, Any]]", argument has type "type[MutableMappingT] | MutableMappingT")
def to_dict(
    df: DataFrame,
    orient: Literal[
        "dict", "list", "series", "split", "tight", "records", "index"
    ] = "dict",
    *,
    into: type[MutableMappingT] | MutableMappingT = dict,  # type: ignore[assignment]
    index: bool = True,
) -> MutableMappingT | list[MutableMappingT]:
    # [FUNCTION DOC]
    # Implementation: convert a DataFrame to a mapping (or a list of mappings for records).
    # Parameters: df is the frame; orient selects the layout (dict, list, series, split, tight, records, index; case-insensitive after validation of the index flag), default dict; into is the mapping class or an empty instance (default dict; a defaultdict must be passed initialized); index (default True) controls whether index entries are included and may only be False for split and tight orientations.
    # Behavior: warns (user warning) if columns are not unique and orient is not tight, since duplicated columns are dropped by key. Standardizes into to a factory. Raises a value error if index is False with another orientation, if orient is index and the frame index is not unique, or if orient is not recognized.
    # Layouts: series maps column to its Series (returned immediately); dict maps column to a mapping of index to value, using the Series conversion; list maps column to a list of native values; split gives index, columns and data lists; tight adds index names and column names; records gives one mapping per row; index maps each index label to a mapping of column to value. Object and extension typed columns have values converted to native Python objects; masked dtypes use the NA marker when producing list values.
    # KNOWN ISSUE: the non-unique-columns warning compares the orientation before it is lowercased, so an upper-case tight orientation would still trigger the warning.
    # Returns a mapping of the requested type, or a list of mappings for records.
    """
    Convert the DataFrame to a dictionary.

    The type of the key-value pairs can be customized with the parameters
    (see below).

    Parameters
    ----------
    orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        Determines the type of the values of the dictionary.

        - 'dict' (default) : dict like {column -> {index -> value}}
        - 'list' : dict like {column -> [values]}
        - 'series' : dict like {column -> Series(values)}
        - 'split' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
        - 'tight' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
          'index_names' -> [index.names], 'column_names' -> [column.names]}
        - 'records' : list like
          [{column -> value}, ... , {column -> value}]
        - 'index' : dict like {index -> {column -> value}}

    into : class, default dict
        The collections.abc.MutableMapping subclass used for all Mappings
        in the return value.  Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

    index : bool, default True
        Whether to include the index item (and index_names item if `orient`
        is 'tight') in the returned dictionary. Can only be ``False``
        when `orient` is 'split' or 'tight'.

        .. versionadded:: 2.0.0

    Returns
    -------
    dict, list or collections.abc.Mapping
        Return a collections.abc.MutableMapping object representing the
        DataFrame. The resulting transformation depends on the `orient` parameter.
    """
    if orient != "tight" and not df.columns.is_unique:
        warnings.warn(
            "DataFrame columns are not unique, some columns will be omitted.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
    # GH16122
    # error: Call to untyped function "standardize_mapping" in typed context
    into_c = com.standardize_mapping(into)  # type: ignore[no-untyped-call]

    #  error: Incompatible types in assignment (expression has type "str",
    # variable has type "Literal['dict', 'list', 'series', 'split', 'tight',
    # 'records', 'index']")
    orient = orient.lower()  # type: ignore[assignment]

    if not index and orient not in ["split", "tight"]:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split' or 'tight'"
        )

    if orient == "series":
        # GH46470 Return quickly if orient series to avoid creating dtype objects
        return into_c((k, v) for k, v in df.items())

    if orient == "dict":
        return into_c((k, v.to_dict(into=into)) for k, v in df.items())

    box_native_indices = [
        i
        for i, col_dtype in enumerate(df.dtypes.values)
        if col_dtype == np.dtype(object) or isinstance(col_dtype, ExtensionDtype)
    ]

    are_all_object_dtype_cols = len(box_native_indices) == len(df.dtypes)

    if orient == "list":
        object_dtype_indices_as_set: set[int] = set(box_native_indices)
        box_na_values = (
            lib.no_default
            if not isinstance(col_dtype, BaseMaskedDtype)
            else libmissing.NA
            for col_dtype in df.dtypes.values
        )
        return into_c(
            (
                k,
                list(map(maybe_box_native, v.to_numpy(na_value=box_na_value)))
                if i in object_dtype_indices_as_set
                else list(map(maybe_box_native, v.to_numpy())),
            )
            for i, (box_na_value, (k, v)) in enumerate(
                zip(box_na_values, df.items(), strict=True)
            )
        )

    elif orient == "split":
        data = list(
            create_data_for_split(df, are_all_object_dtype_cols, box_native_indices)
        )

        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                ("data", data),
            )
        )

    elif orient == "tight":
        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                (
                    "data",
                    [
                        list(map(maybe_box_native, t))
                        for t in df.itertuples(index=False, name=None)
                    ],
                ),
            )
            + ((("index_names", list(df.index.names)),) if index else ())
            + (("column_names", list(df.columns.names)),)
        )

    elif orient == "records":
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            return [
                into_c(zip(columns, map(maybe_box_native, row), strict=True))
                for row in df.itertuples(index=False, name=None)
            ]
        else:
            data = [
                into_c(zip(columns, t, strict=True))
                for t in df.itertuples(index=False, name=None)
            ]
            if box_native_indices:
                object_dtype_indices_as_set = set(box_native_indices)
                object_dtype_cols = {
                    col
                    for i, col in enumerate(df.columns)
                    if i in object_dtype_indices_as_set
                }
                for row in data:
                    for col in object_dtype_cols:
                        row[col] = maybe_box_native(row[col])
            return data  # type: ignore[return-value]

    elif orient == "index":
        if not df.index.is_unique:
            raise ValueError("DataFrame index must be unique for orient='index'.")
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            return into_c(
                (t[0], dict(zip(df.columns, map(maybe_box_native, t[1:]), strict=True)))
                for t in df.itertuples(name=None)
            )
        elif box_native_indices:
            object_dtype_indices_as_set = set(box_native_indices)
            return into_c(
                (
                    t[0],
                    {
                        column: maybe_box_native(v)
                        if i in object_dtype_indices_as_set
                        else v
                        for i, (column, v) in enumerate(
                            zip(columns, t[1:], strict=True)
                        )
                    },
                )
                for t in df.itertuples(name=None)
            )
        else:
            return into_c(
                (t[0], dict(zip(columns, t[1:], strict=True)))
                for t in df.itertuples(name=None)
            )

    else:
        raise ValueError(f"orient '{orient}' not understood")
