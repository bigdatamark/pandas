"""
Table Schema builders

https://specs.frictionlessdata.io/table-schema/
"""

# [MODULE DOC]
# Conversion between pandas objects and the Frictionless Table Schema
# description used by the JSON reader/writer in the table orientation.
# Public contents: the schema version constant; as_json_table_type (dtype to
# schema type name); set_default_names (give unnamed index levels default
# names); convert_pandas_type_to_json_field and
# convert_json_field_to_pandas_type (the two directions of per-column field
# conversion); build_table_schema (schema dictionary for a frame or series);
# parse_table_schema (rebuild a frame from a JSON string holding a schema and
# data). Relationships: used by the sibling JSON reader and writer modules;
# depends on the dtype registry, the JSON decoder and offset parsing.
# No import-time side effects beyond the imports.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

from pandas._config import option_context

from pandas._libs import lib
from pandas._libs.json import ujson_loads
from pandas._libs.tslibs import timezones
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)

from pandas import DataFrame
import pandas.core.common as com

from pandas.tseries.frequencies import to_offset

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        JSONSerializable,
    )

    from pandas import Series
    from pandas.core.indexes.multi import MultiIndex


TABLE_SCHEMA_VERSION = "1.4.0"


def as_json_table_type(x: DtypeObj) -> str:
    # [FUNCTION DOC]
    # Maps a NumPy or extension dtype to a Table Schema type name.
    # Input x is a dtype object. Returns one of: integer (integer dtypes),
    # boolean, number (other numeric dtypes), datetime (datetime64, timezone
    # aware datetimes and periods), duration (timedelta64), string (string
    # dtypes, including object dtype) or any (everything else, e.g.
    # categoricals). Checks run in that order, so integers and booleans are
    # detected before generic numerics. No side effects.
    # KNOWN ISSUE: the docstring table says object maps to str, but the code
    # returns the name string for string-like dtypes (object included).
    """
    Convert a NumPy / pandas type to its corresponding json_table.

    Parameters
    ----------
    x : np.dtype or ExtensionDtype

    Returns
    -------
    str
        the Table Schema data types

    Notes
    -----
    This table shows the relationship between NumPy / pandas dtypes,
    and Table Schema dtypes.

    ==============  =================
    Pandas type     Table Schema type
    ==============  =================
    int64           integer
    float64         number
    bool            boolean
    datetime64[ns]  datetime
    timedelta64[ns] duration
    object          str
    categorical     any
    =============== =================
    """
    if is_integer_dtype(x):
        return "integer"
    elif is_bool_dtype(x):
        return "boolean"
    elif is_numeric_dtype(x):
        return "number"
    elif lib.is_np_dtype(x, "M") or isinstance(x, (DatetimeTZDtype, PeriodDtype)):
        return "datetime"
    elif lib.is_np_dtype(x, "m"):
        return "duration"
    elif is_string_dtype(x):
        return "string"
    else:
        return "any"


def set_default_names(data):
    # [FUNCTION DOC]
    # Ensures the index of a frame or series has names suitable for a schema.
    # If every index level already has a name, returns the input unchanged but
    # emits a warning (attributed to the caller) when the single level is named
    # index, or when a multi-level index has any name beginning with the
    # level prefix, because those names are not round-trippable. Otherwise
    # makes a shallow copy and names the single level index (when its name is
    # empty) or fills missing names of a multi-level index with level-number
    # based names. Returns the (possibly copied) object; the input is never
    # modified.
    # KNOWN ISSUE: the multi-level warning check calls a string method on every
    # level name, so a non-string (e.g. integer) name raises an attribute error.
    """Sets index names to 'index' for regular, or 'level_x' for Multi"""
    if com.all_not_none(*data.index.names):
        nms = data.index.names
        if len(nms) == 1 and data.index.name == "index":
            warnings.warn(
                "Index name of 'index' is not round-trippable.",
                stacklevel=find_stack_level(),
            )
        elif len(nms) > 1 and any(x.startswith("level_") for x in nms):
            warnings.warn(
                "Index names beginning with 'level_' are not round-trippable.",
                stacklevel=find_stack_level(),
            )
        return data

    data = data.copy(deep=False)
    if data.index.nlevels > 1:
        data.index.names = com.fill_missing_names(data.index.names)
    else:
        data.index.name = data.index.name or "index"
    return data


def convert_pandas_type_to_json_field(arr) -> dict[str, JSONSerializable]:
    # [FUNCTION DOC]
    # Builds the Table Schema field descriptor for one Series or Index.
    # Input arr is any object with a dtype and a name. Returns a dictionary
    # with the name (the object's name, or the word values when unnamed) and
    # the schema type from the dtype mapper, plus extras by dtype: categorical
    # adds an enum constraint listing the categories and an ordered flag; period
    # adds the frequency string; timezone-aware datetime adds the zone (UTC, or
    # the zone name when it is a string; omitted when the zone has no string
    # name); any other extension dtype adds its registered name. No side
    # effects.
    dtype = arr.dtype
    name: JSONSerializable
    if arr.name is None:
        name = "values"
    else:
        name = arr.name
    field: dict[str, JSONSerializable] = {
        "name": name,
        "type": as_json_table_type(dtype),
    }

    if isinstance(dtype, CategoricalDtype):
        cats = dtype.categories
        ordered = dtype.ordered

        field["constraints"] = {"enum": list(cats)}
        field["ordered"] = ordered
    elif isinstance(dtype, PeriodDtype):
        field["freq"] = dtype.freq.freqstr
    elif isinstance(dtype, DatetimeTZDtype):
        if timezones.is_utc(dtype.tz):
            field["tz"] = "UTC"
        else:
            zone = timezones.get_timezone(dtype.tz)
            if isinstance(zone, str):
                field["tz"] = zone
    elif isinstance(dtype, ExtensionDtype):
        field["extDtype"] = dtype.name
    return field


def convert_json_field_to_pandas_type(field) -> str | CategoricalDtype:
    # [FUNCTION DOC]
    # Maps a field descriptor (mapping with a type key and optional extra keys)
    # to a dtype specification usable for casting.
    # string: the extension dtype name if present, else none; integer, number,
    # boolean: the extension dtype name if present else int64, float64 or bool;
    # duration: the generic timedelta name; datetime: a timezone-aware
    # nanosecond string if a zone is given, a period spec using the normalized
    # frequency if a frequency is given, else plain nanosecond datetime; any:
    # a categorical dtype when both constraints and ordered keys exist, a
    # registry lookup when an extension dtype name exists, else object.
    # Raises a value error for unknown types.
    # KNOWN ISSUE: for the string type with no extension dtype name the
    # function returns None rather than a string, contradicting the return
    # annotation.
    """
    Converts a JSON field descriptor into its corresponding NumPy / pandas type

    Parameters
    ----------
    field
        A JSON field descriptor

    Returns
    -------
    dtype

    Raises
    ------
    ValueError
        If the type of the provided field is unknown or currently unsupported

    Examples
    --------
    >>> convert_json_field_to_pandas_type({"name": "an_int", "type": "integer"})
    'int64'

    >>> convert_json_field_to_pandas_type(
    ...     {
    ...         "name": "a_categorical",
    ...         "type": "any",
    ...         "constraints": {"enum": ["a", "b", "c"]},
    ...         "ordered": True,
    ...     }
    ... )
    CategoricalDtype(categories=['a', 'b', 'c'], ordered=True, categories_dtype=str)

    >>> convert_json_field_to_pandas_type({"name": "a_datetime", "type": "datetime"})
    'datetime64[ns]'

    >>> convert_json_field_to_pandas_type(
    ...     {"name": "a_datetime_with_tz", "type": "datetime", "tz": "US/Central"}
    ... )
    'datetime64[ns, US/Central]'
    """
    typ = field["type"]
    if typ == "string":
        return field.get("extDtype", None)
    elif typ == "integer":
        return field.get("extDtype", "int64")
    elif typ == "number":
        return field.get("extDtype", "float64")
    elif typ == "boolean":
        return field.get("extDtype", "bool")
    elif typ == "duration":
        return "timedelta64"
    elif typ == "datetime":
        if field.get("tz"):
            return f"datetime64[ns, {field['tz']}]"
        elif field.get("freq"):
            # GH#9586 rename frequency M to ME for offsets
            offset = to_offset(field["freq"])
            freq = PeriodDtype(offset)._freqstr
            # GH#47747 using datetime over period to minimize the change surface
            return f"period[{freq}]"
        else:
            return "datetime64[ns]"
    elif typ == "any":
        if "constraints" in field and "ordered" in field:
            return CategoricalDtype(
                categories=field["constraints"]["enum"], ordered=field["ordered"]
            )
        elif "extDtype" in field:
            return registry.find(field["extDtype"])
        else:
            return "object"

    raise ValueError(f"Unsupported or invalid field type: {typ}")


def build_table_schema(
    data: DataFrame | Series,
    index: bool = True,
    primary_key: bool | None = None,
    version: bool = True,
) -> dict[str, JSONSerializable]:
    # [FUNCTION DOC]
    # Builds a Table Schema dictionary for a frame or series.
    # Inputs: data is the object; index says whether index levels are included
    # as fields (default true); primary_key is None (default; use the index
    # level names when the index is unique), or an explicit value stored as the
    # primary key; version says whether to add the schema version entry
    # (default true).
    # Behavior: when index is exactly true, index names are defaulted first
    # (possibly copying and warning). Fields list index levels first (for a
    # multi-level index, per-level descriptors renamed to the level names; the
    # level values, not the full index, are examined), then columns (or the
    # series itself). Primary key is set to the single index name (as a list)
    # or the list-like of level names when the index is included, unique and
    # primary_key is None; otherwise to the provided value if not None.
    # Returns the dictionary with keys fields, optional primary key, optional
    # version.
    # KNOWN ISSUE: the docstring says primary_key defaults to True and is a
    # bool describing column names, but the signature default is None and the
    # value is stored verbatim.
    # KNOWN ISSUE: for a multi-level index the data index is reassigned through
    # a cast; when index is true but not the boolean true (e.g. truthy value) no
    # default names are applied.
    """
    Create a Table schema from ``data``.

    This method is a utility to generate a JSON-serializable schema
    representation of a pandas Series or DataFrame, compatible with the
    Table Schema specification. It enables structured data to be shared
    and validated in various applications, ensuring consistency and
    interoperability.

    Parameters
    ----------
    data : Series or DataFrame
        The input data for which the table schema is to be created.
    index : bool, default True
        Whether to include ``data.index`` in the schema.
    primary_key : bool or None, default True
        Column names to designate as the primary key.
        The default `None` will set `'primaryKey'` to the index
        level or levels if the index is unique.
    version : bool, default True
        Whether to include a field `pandas_version` with the version
        of pandas that last revised the table schema. This version
        can be different from the installed pandas version.

    Returns
    -------
    dict
        A dictionary representing the Table schema.

    See Also
    --------
    DataFrame.to_json : Convert the object to a JSON string.
    read_json : Convert a JSON string to pandas object.

    Notes
    -----
    See `Table Schema
    <https://pandas.pydata.org/docs/user_guide/io.html#table-schema>`__ for
    conversion types.
    Timedeltas as converted to ISO8601 duration format with
    9 decimal places after the seconds field for nanosecond precision.

    Categoricals are converted to the `any` dtype, and use the `enum` field
    constraint to list the allowed values. The `ordered` attribute is included
    in an `ordered` field.

    Examples
    --------
    >>> from pandas.io.json._table_schema import build_table_schema
    >>> df = pd.DataFrame(
    ...     {'A': [1, 2, 3],
    ...      'B': ['a', 'b', 'c'],
    ...      'C': pd.date_range('2016-01-01', freq='D', periods=3),
    ...      }, index=pd.Index(range(3), name='idx'))
    >>> build_table_schema(df)
    {'fields': \
[{'name': 'idx', 'type': 'integer'}, \
{'name': 'A', 'type': 'integer'}, \
{'name': 'B', 'type': 'string', 'extDtype': 'str'}, \
{'name': 'C', 'type': 'datetime'}], \
'primaryKey': ['idx'], \
'pandas_version': '1.4.0'}
    """
    if index is True:
        data = set_default_names(data)

    schema: dict[str, Any] = {}
    fields = []

    if index:
        if data.index.nlevels > 1:
            data.index = cast("MultiIndex", data.index)
            for level, name in zip(data.index.levels, data.index.names, strict=True):
                new_field = convert_pandas_type_to_json_field(level)
                new_field["name"] = name
                fields.append(new_field)
        else:
            fields.append(convert_pandas_type_to_json_field(data.index))

    if data.ndim > 1:
        for column, s in data.items():
            fields.append(convert_pandas_type_to_json_field(s))
    else:
        fields.append(convert_pandas_type_to_json_field(data))

    schema["fields"] = fields
    if index and data.index.is_unique and primary_key is None:
        if data.index.nlevels == 1:
            schema["primaryKey"] = [data.index.name]
        else:
            schema["primaryKey"] = data.index.names
    elif primary_key is not None:
        schema["primaryKey"] = primary_key

    if version:
        schema["pandas_version"] = TABLE_SCHEMA_VERSION
    return schema


def parse_table_schema(json, precise_float: bool) -> DataFrame:
    # [FUNCTION DOC]
    # Rebuilds a frame from a JSON string holding a schema section and a data
    # section.
    # Inputs: json is the JSON text; precise_float selects the slower, more
    # precise float decoding. Decodes the text, creates a frame with columns in
    # schema field order, converts each column to the dtype implied by its
    # field (inside a temporary option context that does not distinguish NaN
    # from NA), then applies the primary key as the index if present, undoing
    # the default naming: a lone index named index becomes unnamed and
    # multi-level names starting with the level prefix become unnamed.
    # Raises a not-implemented error if any field is a duration. Returns the
    # frame.
    # KNOWN ISSUE: the docstring says timezone data also raises
    # NotImplementedError, but only duration fields do; time zones are
    # supported.
    """
    Builds a DataFrame from a given schema

    Parameters
    ----------
    json :
        A JSON table schema
    precise_float : bool
        Flag controlling precision when decoding string to double values, as
        dictated by ``read_json``

    Returns
    -------
    df : DataFrame

    Raises
    ------
    NotImplementedError
        If the JSON table schema contains either timezone or timedelta data

    Notes
    -----
        Because :func:`DataFrame.to_json` uses the string 'index' to denote a
        name-less :class:`Index`, this function sets the name of the returned
        :class:`DataFrame` to ``None`` when said string is encountered with a
        normal :class:`Index`. For a :class:`MultiIndex`, the same limitation
        applies to any strings beginning with 'level_'. Therefore, an
        :class:`Index` name of 'index'  and :class:`MultiIndex` names starting
        with 'level_' are not supported.

    See Also
    --------
    build_table_schema : Inverse function.
    pandas.read_json
    """
    table = ujson_loads(json, precise_float=precise_float)
    col_order = [field["name"] for field in table["schema"]["fields"]]
    df = DataFrame(table["data"], columns=col_order)[col_order]

    dtypes = {
        field["name"]: convert_json_field_to_pandas_type(field)
        for field in table["schema"]["fields"]
    }

    # No ISO constructor for Timedelta as of yet, so need to raise
    if "timedelta64" in dtypes.values():
        raise NotImplementedError(
            'table="orient" can not yet read ISO-formatted Timedelta data'
        )

    with option_context("future.distinguish_nan_and_na", False):
        df = df.astype(dtypes)

    if "primaryKey" in table["schema"]:
        df = df.set_index(table["schema"]["primaryKey"])
        if len(df.index.names) == 1:
            if df.index.name == "index":
                df.index.name = None
        else:
            df.index.names = [
                None if x.startswith("level_") else x for x in df.index.names
            ]

    return df
