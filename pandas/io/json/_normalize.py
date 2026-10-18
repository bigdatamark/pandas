# ---------------------------------------------------------------------
# JSON normalization routines
# [MODULE DOC]
# Flattening of nested, JSON-like data (dicts, lists of dicts, Series of dicts)
# into tabular form. Public content: the json_normalize function (exported as a
# top-level pandas function), the nested_to_record helper that flattens dicts
# with a configurable depth limit, and convert_to_line_delimits which turns a
# JSON list string into line-delimited JSON. Private helpers implement a fast
# path for the plain "no options" case and validation of the meta argument.
# Relationships: used by the JSON reader/writer in the sibling json package and
# by DataFrame construction; depends on a Cython writer routine for line
# conversion and imports the top-level pandas package for null checks.
# No import-time side effects beyond the imports.
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import copy
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    overload,
)

import numpy as np

from pandas._libs.writers import convert_json_to_lines
from pandas.util._decorators import set_module

from pandas.core.dtypes.common import is_scalar

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas._typing import (
        IgnoreRaise,
        Scalar,
    )


def convert_to_line_delimits(s: str) -> str:
    # [FUNCTION DOC]
    # HELPER: used by the JSON writer to emit line-delimited output.
    # Takes a JSON string s. If the string looks like a JSON list, strips the
    # outer brackets and delegates to the compiled line converter, returning
    # the line-delimited string. Otherwise returns the input unchanged.
    # KNOWN ISSUE: the guard is written as "not first-char-is-open-bracket and
    # last-char-is-close-bracket" due to operator precedence, so it only
    # returns early for strings that do not start with an opening bracket but
    # do end with a closing one; the intended check (list must start AND end
    # with brackets) is not what is evaluated. Empty strings raise an index
    # error.
    """
    Helper function that converts JSON lists to line delimited JSON.
    """
    # Determine we have a JSON list to turn to lines otherwise just return the
    # json object, only lists can
    if not s[0] == "[" and s[-1] == "]":
        return s
    s = s[1:-1]

    return convert_json_to_lines(s)


@overload
def nested_to_record(
    ds: dict,
    prefix: str = ...,
    sep: str = ...,
    level: int = ...,
    max_level: int | None = ...,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Overload stub: a single dict input yields a single flat dict. See the
    # implementation below.
    ...


@overload
def nested_to_record(
    ds: list[dict],
    prefix: str = ...,
    sep: str = ...,
    level: int = ...,
    max_level: int | None = ...,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Overload stub: a list of dicts yields a list of flat dicts. See the
    # implementation below.
    ...


def nested_to_record(
    ds: dict | list[dict],
    prefix: str = "",
    sep: str = ".",
    level: int = 0,
    max_level: int | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Flattens nested dicts into flat dicts whose keys are the path of nested
    # keys joined by the separator. Does not extract sub-records.
    # Inputs: ds is a dict or a list of dicts; prefix is the key path built so
    # far (used by recursion; ignored at level zero); sep is the joiner for key
    # parts; level is the current depth (recursion bookkeeping, callers leave
    # it at zero); max_level is the maximum depth to flatten, or None for
    # unlimited. Non-string keys are converted to strings.
    # Returns a dict if a dict was given, otherwise a list of new dicts. Input
    # is deep-copied, so the argument is not mutated. Only dict values are
    # recursed into; values at or beyond max_level are left as-is (but renamed
    # with the prefix when below the top level). Dict values are removed and
    # replaced by their flattened entries.
    # KNOWN ISSUE: the docstring describes the level parameter as the number of
    # levels in the json string, but it is only the current recursion depth.
    """
    A simplified json_normalize

    Converts a nested dict into a flat dict ("record"), unlike json_normalize,
    it does not attempt to extract a subset of the data.

    Parameters
    ----------
    ds : dict or list of dicts
    prefix: the prefix, optional, default: ""
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    level: int, optional, default: 0
        The number of levels in the json string.

    max_level: int, optional, default: None
        The max depth to normalize.

    Returns
    -------
    d - dict or list of dicts, matching `ds`

    Examples
    --------
    >>> nested_to_record(
    ...     dict(flat1=1, dict1=dict(c=1, d=2), nested=dict(e=dict(c=1, d=2), d=2))
    ... )
    {\
'flat1': 1, \
'dict1.c': 1, \
'dict1.d': 2, \
'nested.e.c': 1, \
'nested.e.d': 2, \
'nested.d': 2\
}
    """
    singleton = False
    if isinstance(ds, dict):
        ds = [ds]
        singleton = True
    new_ds = []
    for d in ds:
        new_d = copy.deepcopy(d)
        for k, v in d.items():
            # each key gets renamed with prefix
            if not isinstance(k, str):
                k = str(k)
            if level == 0:
                newkey = k
            else:
                newkey = prefix + sep + k

            # flatten if type is dict and
            # current dict level  < maximum level provided and
            # only dicts gets recurse-flattened
            # only at level>1 do we rename the rest of the keys
            if not isinstance(v, dict) or (
                max_level is not None and level >= max_level
            ):
                if level != 0:  # so we skip copying for top level, common case
                    v = new_d.pop(k)
                    new_d[newkey] = v
                continue

            v = new_d.pop(k)
            new_d.update(nested_to_record(v, newkey, sep, level + 1, max_level))
        new_ds.append(new_d)

    if singleton:
        return new_ds[0]
    return new_ds


def _normalize_json(
    data: Any,
    key_string: str,
    normalized_dict: dict[str, Any],
    separator: str,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: recursive worker for the fast path of the simple JSON normalizer.
    # Inputs: data is the current value (a dict is recursed into, anything else
    # is a leaf); key_string is the joined key path so far (empty at the top);
    # normalized_dict is the accumulator that is mutated in place; separator
    # joins key parts. For dict input, each child key is appended to the path
    # (a leading separator is stripped when the path was empty). Leaves are
    # stored in the accumulator under the full path. Returns the accumulator.
    # Keys are formatted via string interpolation, so non-string keys work.
    """
    Main recursive function
    Designed for the most basic use case of pd.json_normalize(data)
    intended as a performance improvement, see #15621

    Parameters
    ----------
    data : Any
        Type dependent on types contained within nested Json
    key_string : str
        New key (with separator(s) in) for data
    normalized_dict : dict
        The new normalized/flattened Json dict
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{key_string}{separator}{key}"

            if not key_string:
                new_key = new_key.removeprefix(separator)

            _normalize_json(
                data=value,
                key_string=new_key,
                normalized_dict=normalized_dict,
                separator=separator,
            )
    else:
        normalized_dict[key_string] = data
    return normalized_dict


def _normalize_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: flattens one record for the simple fast path.
    # Takes a dict and a separator. Returns a new dict in which all
    # non-dict top-level entries come first (original order), followed by the
    # fully flattened entries from dict-valued keys. This ordering matches the
    # column order of the general algorithm. Input is not mutated.
    # KNOWN ISSUE: the docstring says the input may be a list of dicts and that
    # the return matches a name that does not exist as a parameter; the code
    # only accepts a single dict.
    """
    Order the top level keys and then recursively go to depth

    Parameters
    ----------
    data : dict or list of dicts
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    dict or list of dicts, matching `normalized_json_object`
    """
    top_dict_ = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested_dict_ = _normalize_json(
        data={k: v for k, v in data.items() if isinstance(v, dict)},
        key_string="",
        normalized_dict={},
        separator=separator,
    )
    return {**top_dict_, **nested_dict_}


def _simple_json_normalize(
    ds: dict | list[dict],
    sep: str = ".",
) -> dict | list[dict] | Any:
    # [FUNCTION DOC]
    # HELPER: optimized implementation of normalization when no record path,
    # meta, prefixes or max level are requested.
    # Inputs: ds is a dict or list of dicts; sep is the key joiner. A dict is
    # flattened via the ordered helper; a list is processed element-wise
    # (recursively) and a list of flat dicts is returned; any other type yields
    # an empty dict. The caller wraps the result in a DataFrame.
    # KNOWN ISSUE: the docstring lists a frame return that does not exist; the
    # function returns dicts or lists of dicts.
    """
    An optimized basic json_normalize

    Converts a nested dict into a flat dict ("record"), unlike
    json_normalize and nested_to_record it doesn't do anything clever.
    But for the most basic use cases it enhances performance.
    E.g. pd.json_normalize(data)

    Parameters
    ----------
    ds : dict or list of dicts
    sep : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    frame : DataFrame
    d - dict or list of dicts, matching `normalized_json_object`

    Examples
    --------
    >>> _simple_json_normalize(
    ...     {
    ...         "flat1": 1,
    ...         "dict1": {"c": 1, "d": 2},
    ...         "nested": {"e": {"c": 1, "d": 2}, "d": 2},
    ...     }
    ... )
    {\
'flat1': 1, \
'dict1.c': 1, \
'dict1.d': 2, \
'nested.e.c': 1, \
'nested.e.d': 2, \
'nested.d': 2\
}

    """
    normalized_json_object = {}
    # expect a dictionary, as most jsons are. However, lists are perfectly valid
    if isinstance(ds, dict):
        normalized_json_object = _normalize_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalized_json_list = [_simple_json_normalize(row, sep=sep) for row in ds]
        return normalized_json_list
    return normalized_json_object


def _validate_meta(meta: str | list[str | list[str]] | None) -> None:
    # [FUNCTION DOC]
    # HELPER: argument validation for the public normalizer.
    # Accepts nothing, a single string, or a list whose items are strings or
    # lists of strings. Returns nothing. Raises a type error naming the
    # offending element when a nested path contains a non-string, or when a
    # top-level item is neither a string nor a list. No mutation.
    """
    Validate that meta parameter contains only strings or lists of strings.
    Parameters
    ----------
    meta : str or list of str or list of list of str or None
        The meta parameter to validate.
    Raises
    ------
    TypeError
        If meta contains elements that are not strings or lists of strings.
    """
    if meta is None:
        return
    if isinstance(meta, str):
        return
    for item in meta:
        if isinstance(item, list):
            for subitem in item:
                if not isinstance(subitem, str):
                    raise TypeError(
                        "All elements in nested meta paths must be strings. "
                        f"Found {type(subitem).__name__}: {subitem!r}"
                    )
        elif not isinstance(item, str):
            raise TypeError(
                "All elements in 'meta' must be strings or lists of strings. "
                f"Found {type(item).__name__}: {item!r}"
            )


@set_module("pandas")
def json_normalize(
    data: dict | list[dict] | Series,
    record_path: str | list | None = None,
    meta: str | list[str | list[str]] | None = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: IgnoreRaise = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> DataFrame:
    # [FUNCTION DOC]
    # Public entry point (exported as a top-level pandas function) that turns
    # JSON-like data into a flat DataFrame.
    # Inputs: data is a dict, an iterable of dicts (generators are consumed),
    # or a Series of dicts (its index is preserved); record_path is a key or
    # list of keys leading to the list of records to expand (default none: the
    # data are the records); meta is a key or list of keys/key-paths to carry
    # onto every record row; meta_prefix and record_prefix are optional strings
    # prepended to meta column names and record column names; errors is raise
    # or ignore, controlling whether missing meta keys raise a key error or
    # become missing values; sep joins nested key names; max_level limits the
    # flattening depth (None means all levels).
    # Behavior: validates meta first. An empty list gives an empty frame. A
    # single dict is wrapped in a list. Other iterables are materialized and
    # each item must be a dict or a scalar missing value (which becomes an empty
    # dict). With no options at all, the fast simple path is used. Without a
    # record path, records are flattened with the nested-record helper (when
    # any dict value is present) and optionally have columns prefixed. With a
    # record path, the path is walked recursively, record dicts are flattened,
    # meta values are collected per parent and repeated for each of its
    # records, and the result index is repeated per record when the input was a
    # Series.
    # Raises: a type error for invalid meta, for non-dict non-missing items, or
    # when a record path target is neither a list nor null; a key error for
    # missing record path keys (always) or missing meta keys (unless errors is
    # ignore); a value error when a meta column name collides with a record
    # column; a not-implemented error when data is not a dict or non-string
    # iterable.
    # KNOWN ISSUE: the any-dict check in the no-record-path branch builds a
    # generator of lists for each record, and a non-empty list is truthy
    # regardless of its contents, so flattening is applied whenever any record
    # has at least one entry, even if no dict values exist (harmless but the
    # test is ineffective).
    """
    Normalize semi-structured JSON data into a flat table.

    This method is designed to transform semi-structured JSON data, such as nested
    dictionaries or lists, into a flat table. This is particularly useful when
    handling JSON-like data structures that contain deeply nested fields.

    Parameters
    ----------
    data : dict, list of dicts, or Series of dicts
        Unserialized JSON objects.
    record_path : str or list of str, default None
        Path in each object to list of records. If not passed, data will be
        assumed to be an array of records.
    meta : list of paths (str or list of str), default None
        Fields to use as metadata for each record in resulting table.
    meta_prefix : str, default None
        String to prefix records with dotted path, e.g. foo.bar.field if
        meta is ['foo', 'bar'].
    record_prefix : str, default None
        String to prefix records with dotted path, e.g. foo.bar.field if
        path to records is ['foo', 'bar'].
    errors : {'raise', 'ignore'}, default 'raise'
        Configures error handling.

        * 'ignore' : will ignore KeyError if keys listed in meta are not
          always present.
        * 'raise' : will raise KeyError if keys listed in meta are not
          always present.
    sep : str, default '.'
        Nested records will generate names separated by sep.
        e.g., for sep='.', {'foo': {'bar': 0}} -> foo.bar.
    max_level : int, default None
        Max number of levels(depth of dict) to normalize.
        if None, normalizes all levels.

    Returns
    -------
    DataFrame
        The normalized data, represented as a pandas DataFrame.

    See Also
    --------
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous tabular data.
    Series : One-dimensional ndarray with axis labels (including time series).

    Examples
    --------
    >>> data = [
    ...     {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    ...     {"name": {"given": "Mark", "family": "Regner"}},
    ...     {"id": 2, "name": "Faye Raker"},
    ... ]
    >>> pd.json_normalize(data)
        id name.first name.last name.given name.family        name
    0  1.0     Coleen      Volk        NaN         NaN         NaN
    1  NaN        NaN       NaN       Mark      Regner         NaN
    2  2.0        NaN       NaN        NaN         NaN  Faye Raker

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=0)
        id        name                        fitness
    0  1.0   Cole Volk  {'height': 130, 'weight': 60}
    1  NaN    Mark Reg  {'height': 130, 'weight': 60}
    2  2.0  Faye Raker  {'height': 130, 'weight': 60}

    Normalizes nested data up to level 1.

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> pd.json_normalize(data, max_level=1)
        id        name  fitness.height  fitness.weight
    0  1.0   Cole Volk             130              60
    1  NaN    Mark Reg             130              60
    2  2.0  Faye Raker             130              60

    >>> data = [
    ...     {
    ...         "id": 1,
    ...         "name": "Cole Volk",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    ...     {
    ...         "id": 2,
    ...         "name": "Faye Raker",
    ...         "fitness": {"height": 130, "weight": 60},
    ...     },
    ... ]
    >>> series = pd.Series(data, index=pd.Index(["a", "b", "c"]))
    >>> pd.json_normalize(series)
        id        name  fitness.height  fitness.weight
    a  1.0   Cole Volk             130              60
    b  NaN    Mark Reg             130              60
    c  2.0  Faye Raker             130              60

    >>> data = [
    ...     {
    ...         "state": "Florida",
    ...         "shortname": "FL",
    ...         "info": {"governor": "Rick Scott"},
    ...         "counties": [
    ...             {"name": "Dade", "population": 12345},
    ...             {"name": "Broward", "population": 40000},
    ...             {"name": "Palm Beach", "population": 60000},
    ...         ],
    ...     },
    ...     {
    ...         "state": "Ohio",
    ...         "shortname": "OH",
    ...         "info": {"governor": "John Kasich"},
    ...         "counties": [
    ...             {"name": "Summit", "population": 1234},
    ...             {"name": "Cuyahoga", "population": 1337},
    ...         ],
    ...     },
    ... ]
    >>> result = pd.json_normalize(
    ...     data, "counties", ["state", "shortname", ["info", "governor"]]
    ... )
    >>> result
             name  population    state shortname info.governor
    0        Dade       12345   Florida    FL    Rick Scott
    1     Broward       40000   Florida    FL    Rick Scott
    2  Palm Beach       60000   Florida    FL    Rick Scott
    3      Summit        1234   Ohio       OH    John Kasich
    4    Cuyahoga        1337   Ohio       OH    John Kasich

    >>> data = {"A": [1, 2]}
    >>> pd.json_normalize(data, "A", record_prefix="Prefix.")
        Prefix.0
    0          1
    1          2

    Returns normalized data with columns prefixed with the given string.
    """
    _validate_meta(meta)

    def _pull_field(
        js: dict[str, Any], spec: list | str, extract_record: bool = False
    ) -> Scalar | Iterable:
        # [FUNCTION DOC]
        # HELPER: nested function of the public normalizer that looks up a
        # value in one record.
        # js is the record; spec is a single key or a list of keys describing a
        # path; extract_record marks lookups of the record path itself. A
        # missing key (or a None encountered midway) raises a key error with an
        # explanatory message when extract_record is set; otherwise, if errors
        # is ignore in the enclosing scope, returns a missing value, else raises
        # a key error suggesting that option. Returns the value found.
        """Internal function to pull field"""
        result = js
        try:
            if isinstance(spec, list):
                for field in spec:
                    if result is None:
                        raise KeyError(field)
                    result = result[field]
            else:
                result = result[spec]
        except KeyError as e:
            if extract_record:
                raise KeyError(
                    f"Key {e} not found. If specifying a record_path, all elements of "
                    f"data should have the path."
                ) from e
            if errors == "ignore":
                return np.nan
            else:
                raise KeyError(
                    f"Key {e} not found. To replace missing values of {e} with "
                    f"np.nan, pass in errors='ignore'"
                ) from e

        return result

    def _pull_records(js: dict[str, Any], spec: list | str) -> list:
        # [FUNCTION DOC]
        # HELPER: nested function of the public normalizer that fetches the
        # list of records at a path. Uses the field puller in record mode.
        # Returns the list found; a null value gives an empty list; any other
        # non-list value raises a type error stating the type and path.
        """
        Internal function to pull field for records, and similar to
        _pull_field, but require to return list. And will raise error
        if has non iterable value.
        """
        result = _pull_field(js, spec, extract_record=True)

        # GH 31507 GH 30145, GH 26284 if result is not list, raise TypeError if not
        # null, otherwise return an empty list
        if not isinstance(result, list):
            if pd.isnull(result):
                result = []
            else:
                raise TypeError(
                    f"Path must contain list or null, "
                    f"but got {type(result).__name__} at {spec!r}"
                )
        return result

    if isinstance(data, Series):
        index = data.index
    else:
        index = None

    if isinstance(data, list) and not data:
        return DataFrame()
    elif isinstance(data, dict):
        # A bit of a hackjob
        data = [data]
    elif isinstance(data, abc.Iterable) and not isinstance(data, str):
        # GH35923 Fix pd.json_normalize to not skip the first element of a
        # generator input
        data = list(data)
        for i, item in enumerate(data):
            if isinstance(item, dict):
                continue
            if is_scalar(item) and pd.isna(item):
                data[i] = {}
            else:
                msg = (
                    "All items in data must be of type dict or NA-like, "
                    f"found {type(item).__name__}"
                )
                raise TypeError(msg)
    else:
        raise NotImplementedError

    # check to see if a simple recursive function is possible to
    # improve performance (see #15621) but only for cases such
    # as pd.Dataframe(data) or pd.Dataframe(data, sep)
    if (
        record_path is None
        and meta is None
        and meta_prefix is None
        and record_prefix is None
        and max_level is None
    ):
        return DataFrame(_simple_json_normalize(data, sep=sep), index=index)

    if record_path is None:
        if any([isinstance(x, dict) for x in y.values()] for y in data):
            # naive normalization, this is idempotent for flat records
            # and potentially will inflate the data considerably for
            # deeply nested structures:
            #  {VeryLong: { b: 1,c:2}} -> {VeryLong.b:1 ,VeryLong.c:@}
            #
            # TODO: handle record value which are lists, at least error
            #       reasonably
            data = nested_to_record(data, sep=sep, max_level=max_level)
        result = DataFrame(data, index=index)
        if record_prefix is not None:
            result = result.rename(columns=lambda x: f"{record_prefix}{x}")
        return result
    elif not isinstance(record_path, list):
        record_path = [record_path]

    if meta is None:
        meta = []
    elif not isinstance(meta, list):
        meta = [meta]

    _meta = [m if isinstance(m, list) else [m] for m in meta]

    # Disastrously inefficient for now
    records: list = []
    lengths = []

    meta_vals: DefaultDict = defaultdict(list)
    meta_keys = [sep.join(val) for val in _meta]

    def _recursive_extract(data, path, seen_meta, level: int = 0) -> None:
        # [FUNCTION DOC]
        # HELPER: nested function of the public normalizer that walks the record
        # path, mutating the enclosing lists and mappings (records, lengths,
        # meta values).
        # data is a dict or list of dicts at the current depth; path is the
        # remaining record path; seen_meta stores meta values captured on the
        # way down, keyed by joined meta name; level is the depth. While more
        # than one path element remains, captures meta values whose path ends at
        # this depth and recurses into the next path element of each object. At
        # the last element, pulls the records, flattens dict records, appends
        # the record count to the lengths list, collects meta values for each
        # meta key (from seen_meta if defined at a shallower depth, else pulled
        # from the object), and extends the records list. Returns nothing.
        if isinstance(data, dict):
            data = [data]
        if len(path) > 1:
            for obj in data:
                for val, key in zip(_meta, meta_keys, strict=True):
                    if level + 1 == len(val):
                        seen_meta[key] = _pull_field(obj, val[-1])

                _recursive_extract(obj[path[0]], path[1:], seen_meta, level=level + 1)
        else:
            for obj in data:
                recs = _pull_records(obj, path[0])
                recs = [
                    nested_to_record(r, sep=sep, max_level=max_level)
                    if isinstance(r, dict)
                    else r
                    for r in recs
                ]

                # For repeating the metadata later
                lengths.append(len(recs))
                for val, key in zip(_meta, meta_keys, strict=True):
                    if level + 1 > len(val):
                        meta_val = seen_meta[key]
                    else:
                        meta_val = _pull_field(obj, val[level:])
                    meta_vals[key].append(meta_val)
                records.extend(recs)

    _recursive_extract(data, record_path, {}, level=0)

    result = DataFrame(records)

    if record_prefix is not None:
        result = result.rename(columns=lambda x: f"{record_prefix}{x}")

    # Data types, a problem
    for k, v in meta_vals.items():
        if meta_prefix is not None:
            k = meta_prefix + k

        if k in result:
            raise ValueError(
                f"Conflicting metadata name {k}, need distinguishing prefix "
            )
        # GH 37782

        values = np.array(v, dtype=object)

        if values.ndim > 1:
            # GH 37782
            values = np.empty((len(v),), dtype=object)
            for i, val in enumerate(v):
                values[i] = val

        result[k] = values.repeat(lengths)
    if index is not None:
        result.index = index.repeat(lengths)
    return result
