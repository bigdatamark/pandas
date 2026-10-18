# [MODULE DOC]
# HELPER: internal utilities shared by the I/O readers for converting pyarrow
# tables into data frames with the right dtypes. Contents: the main entry
# point that converts a table honouring the dtype backend choice, a mapping
# builder from arrow types to pandas nullable dtypes, a mapper for arrow string
# types to the NaN-based string dtype, and post-processing helpers that apply
# user dtypes, convert string dtypes back to object when the string-inference
# option is off, and normalise pytz timezones to zoneinfo or fixed offsets.
# Used by csv, parquet, orc, feather, json and similar readers.
# Import-time side effects: looks up the optional pytz package (ignoring its
# absence) and stores it in a module level variable.

from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
import zoneinfo

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
    pa_version_under18p0,
    pa_version_under19p0,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.common import pandas_dtype

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )

    import pyarrow

    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
    )


pytz = import_optional_dependency("pytz", errors="ignore")


def _arrow_dtype_mapping() -> dict:
    # [FUNCTION DOC]
    # HELPER: builds the arrow-type to pandas-nullable-dtype lookup used by the
    # numpy-nullable backend. Returns a dict keyed by pyarrow type instances
    # (signed and unsigned 8-64 bit integers, bool, string, large string,
    # 32/64 bit floats) with the matching nullable pandas dtype instances as
    # values. Requires pyarrow (import error otherwise).
    # KNOWN ISSUE: the string type key appears twice in the literal; the second
    # entry is redundant.
    pa = import_optional_dependency("pyarrow")
    return {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.string(): pd.StringDtype(),
        pa.float32(): pd.Float32Dtype(),
        pa.float64(): pd.Float64Dtype(),
        pa.string(): pd.StringDtype(),
        pa.large_string(): pd.StringDtype(),
    }


def _arrow_string_types_mapper() -> Callable:
    # [FUNCTION DOC]
    # HELPER: builds the types mapper used when string inference is on with
    # older pyarrow. Returns the lookup method of a mapping from arrow string
    # and large string types (plus string view when pyarrow is at least version
    # 18) to the string dtype using NaN as missing value; unmapped types yield
    # none. Requires pyarrow.
    pa = import_optional_dependency("pyarrow")

    mapping = {
        pa.string(): pd.StringDtype(na_value=np.nan),
        pa.large_string(): pd.StringDtype(na_value=np.nan),
    }
    if not pa_version_under18p0:
        mapping[pa.string_view()] = pd.StringDtype(na_value=np.nan)

    return mapping.get


def arrow_table_to_pandas(
    table: pyarrow.Table,
    dtype_backend: DtypeBackend | Literal["numpy"] | lib.NoDefault = lib.no_default,
    null_to_int64: bool = False,
    to_pandas_kwargs: dict | None = None,
    dtype: DtypeArg | None = None,
    names: Sequence[Hashable] | None = None,
) -> pd.DataFrame:
    # [FUNCTION DOC]
    # Convert a pyarrow table to a pandas data frame with the requested dtype
    # backend.
    # Inputs: table is the pyarrow table; dtype_backend is numpy-nullable,
    # pyarrow, numpy or the no-default sentinel (default); null_to_int64
    # (default false) additionally maps arrow null columns to the nullable
    # 64-bit integer dtype under the nullable backend (to mimic the csv
    # parser); to_pandas_kwargs (default none, treated as empty) are forwarded
    # to the table conversion; dtype is an optional user dtype (single or
    # mapping) to apply afterwards; names optionally gives column names used
    # when a mapping dtype is applied.
    # Behavior: picks a types mapper (nullable dtypes lookup; arrow dtype class;
    # when the string-dtype option is on, the NaN string mapper for pyarrow
    # older than 19, or integer-only nullable mapping when a dtype was
    # supplied; for the default/numpy backends, an integer-only nullable
    # mapping when a dtype was supplied so large ints are not lossily turned
    # into floats, otherwise no mapper), converts, then post-converts dtypes and
    # normalises timezones.
    # Raises: not-implemented error for an unrecognised backend.
    pa = import_optional_dependency("pyarrow")

    to_pandas_kwargs = {} if to_pandas_kwargs is None else to_pandas_kwargs

    types_mapper: type[pd.ArrowDtype] | None | Callable
    if dtype_backend == "numpy_nullable":
        mapping = _arrow_dtype_mapping()
        if null_to_int64:
            # Modify the default mapping to also map null to Int64
            # (to match other engines - only for CSV parser)
            mapping[pa.null()] = pd.Int64Dtype()
        types_mapper = mapping.get
    elif dtype_backend == "pyarrow":
        types_mapper = pd.ArrowDtype
    elif using_string_dtype():
        if pa_version_under19p0:
            types_mapper = _arrow_string_types_mapper()
        elif dtype is not None:
            # GH#56136 Avoid lossy conversion to float64
            # We'll convert to numpy below if
            types_mapper = {
                pa.int8(): pd.Int8Dtype(),
                pa.int16(): pd.Int16Dtype(),
                pa.int32(): pd.Int32Dtype(),
                pa.int64(): pd.Int64Dtype(),
            }.get
        else:
            types_mapper = None
    elif dtype_backend is lib.no_default or dtype_backend == "numpy":
        if dtype is not None:
            # GH#56136 Avoid lossy conversion to float64
            # We'll convert to numpy below if
            types_mapper = {
                pa.int8(): pd.Int8Dtype(),
                pa.int16(): pd.Int16Dtype(),
                pa.int32(): pd.Int32Dtype(),
                pa.int64(): pd.Int64Dtype(),
            }.get
        else:
            types_mapper = None
    else:
        raise NotImplementedError

    df = table.to_pandas(types_mapper=types_mapper, **to_pandas_kwargs)
    df = _post_convert_dtypes(df, dtype_backend, dtype, names)
    df = _normalize_timezone_dtypes(df)
    return df


def _post_convert_dtypes(
    df: pd.DataFrame,
    dtype_backend: DtypeBackend | Literal["numpy"] | lib.NoDefault,
    dtype: DtypeArg | None,
    names: Sequence[Hashable] | None,
) -> pd.DataFrame:
    # [FUNCTION DOC]
    # HELPER: post-processing step of the table conversion.
    # Inputs: df is the converted frame (mutated and returned); dtype_backend
    # as above; dtype is the optional user dtype (a mapping is also mutated by
    # adding entries); names optionally replaces the frame's columns before
    # mapping lookups.
    # Behavior: for the default/numpy backends with a user dtype, mapping keys
    # absent from the frame are ignored, any column the user did not mention
    # that was converted to a nullable integer dtype is converted back to its
    # numpy dtype, then the frame is cast to the dtype. A type error from the
    # cast is re-raised as a value error. Then, when the string-dtype option is
    # off, the requested dtype is not the plain string alias and the backend is
    # default/numpy, string-dtype columns, the index and the columns are
    # converted back to object dtype.
    # Returns the processed frame.
    if dtype is not None and (
        dtype_backend is lib.no_default or dtype_backend == "numpy"
    ):
        # GH#56136 apply any user-provided dtype, and convert any IntegerDtype
        #  columns the user didn't explicitly ask for.
        if isinstance(dtype, dict):
            if names is not None:
                df.columns = names

            cmp_dtypes = {
                pd.Int8Dtype(),
                pd.Int16Dtype(),
                pd.Int32Dtype(),
                pd.Int64Dtype(),
            }
            for col in df.columns:
                if col not in dtype and df[col].dtype in cmp_dtypes:
                    # Any key that the user didn't explicitly specify
                    #  that got converted to IntegerDtype now gets converted
                    #  to numpy dtype.
                    dtype[col] = df[col].dtype.numpy_dtype

            # Ignore non-existent columns from dtype mapping
            # like other parsers do
            dtype = {
                key: pandas_dtype(dtype[key]) for key in dtype if key in df.columns
            }

        else:
            dtype = pandas_dtype(dtype)

        try:
            df = df.astype(dtype)
        except TypeError as err:
            # GH#44901 reraise to keep api consistent
            raise ValueError(str(err)) from err

    if (
        not using_string_dtype()
        and dtype != "str"
        and (dtype_backend is lib.no_default or dtype_backend == "numpy")
    ):
        # Convert any StringDtype columns back to object dtype (pyarrow always
        # uses string dtype even when the infer_string option is False)
        for i in range(len(df.columns)):
            new_col = _maybe_convert_string_to_object(df.iloc[:, i])
            if new_col is not None:
                df.isetitem(i, new_col)

        new_idx = _maybe_convert_string_index_to_object(df.index)
        if new_idx is not None:
            df.index = new_idx
        new_cols = _maybe_convert_string_index_to_object(df.columns)
        if new_cols is not None:
            df.columns = new_cols

    return df


def _maybe_convert_string_to_object(
    data: pd.Series | pd.Index,
) -> pd.Series | pd.Index | None:
    # [FUNCTION DOC]
    # HELPER: converts NaN-based string dtype data (or categorical data whose
    # categories have that dtype) to object dtype. Input: a series or index.
    # Returns the converted object, or none when no conversion is needed.
    # String data is cast to object and then missing values are filled with
    # the none sentinel; categoricals get object categories with ordering
    # preserved.
    if isinstance(data.dtype, pd.StringDtype) and data.dtype.na_value is np.nan:
        return data.astype("object").fillna(None)
    elif isinstance(data.dtype, pd.CategoricalDtype):
        cat_dtype = data.dtype.categories.dtype
        if isinstance(cat_dtype, pd.StringDtype) and cat_dtype.na_value is np.nan:
            cat_dtype = pd.CategoricalDtype(
                categories=data.dtype.categories.astype("object"),
                ordered=data.dtype.ordered,
            )
            return data.astype(cat_dtype)

    # no conversion needed
    return None


def _maybe_convert_string_index_to_object(index: pd.Index) -> pd.Index | None:
    # [FUNCTION DOC]
    # HELPER: index version of the string-to-object conversion. Input: an
    # index or multi-index. For a multi-index, if any level has the NaN-based
    # string dtype, returns a new multi-index with each level converted where
    # needed; otherwise none. For a flat index returns the converted index or
    # none if no conversion is needed.
    if isinstance(index, pd.MultiIndex):
        if any(
            isinstance(level.dtype, pd.StringDtype) and level.dtype.na_value is np.nan
            for level in index.levels
        ):
            new_levels = []
            for level in index.levels:
                new_level = _maybe_convert_string_to_object(level)
                if new_level is not None:
                    new_levels.append(new_level)
                else:
                    new_levels.append(level)
            return index.set_levels(new_levels)
        return None

    else:
        return cast("pd.Index | None", _maybe_convert_string_to_object(index))


def _normalize_pytz_timezone(tz: dt.tzinfo) -> dt.tzinfo:
    # [FUNCTION DOC]
    # HELPER: maps a pytz timezone to a standard-library one. Input: a tzinfo.
    # Returns the input unchanged if its class is not from pytz; the standard
    # UTC timezone for UTC; a zoneinfo timezone for a named zone when available
    # (lookup failures are swallowed); a fixed-offset timezone for pytz fixed
    # offsets; otherwise the input.
    """
    If the input tz is a pytz timezone, attempt to convert it to "default"
    tzinfo object (zoneinfo or datetime.timezone).
    """
    if not type(tz).__module__.startswith("pytz"):
        # isinstance(col.dtype.tz, pytz.BaseTzInfo) does not included
        # fixed offsets
        return tz

    if timezones.is_utc(tz):
        return dt.timezone.utc

    if tz.zone is not None:  # type: ignore[attr-defined]
        try:
            return zoneinfo.ZoneInfo(tz.zone)  # type: ignore[attr-defined]
        except Exception:
            # some pytz timezones might not be available for zoneinfo
            pass

    if timezones.is_fixed_offset(tz):
        # Convert pytz fixed offset to datetime.timezone
        try:
            offset = tz.utcoffset(None)
            if offset is not None:
                return dt.timezone(offset)
        except Exception:
            pass

    return tz


def _normalize_timezone_index(index: pd.Index) -> pd.Index:
    # [FUNCTION DOC]
    # HELPER: normalises pytz timezones on a datetime index. Input: an index or
    # multi-index. For a multi-index, levels with timezone-aware dtype are
    # normalised recursively and returned as a new multi-index; for a flat
    # timezone-aware index, converts to the normalised timezone only if it
    # differs. Otherwise returns the index unchanged.
    if isinstance(index, pd.MultiIndex):
        if any(isinstance(level.dtype, pd.DatetimeTZDtype) for level in index.levels):
            levels = [_normalize_timezone_index(level) for level in index.levels]
            return index.set_levels(levels)

        return index

    if isinstance(index.dtype, pd.DatetimeTZDtype):
        normalized_tz = _normalize_pytz_timezone(index.dtype.tz)
        if normalized_tz is not index.dtype.tz:
            return index.tz_convert(normalized_tz)  # type: ignore[attr-defined]

    return index


def _normalize_timezone_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # [FUNCTION DOC]
    # HELPER: when pytz is installed, replaces pytz timezones on timezone-aware
    # columns, the index and the columns of the given frame with standard
    # timezones. Input: the frame (mutated in place by column replacement and
    # index/column assignment). Returns the same frame. Does nothing when pytz
    # is unavailable.
    """
    PyArrow uses pytz by default for timezones, but pandas uses
    zoneinfo / datetime.timezone since pandas 3.0.

    TODO: Starting with pyarrow 25, it will use zoneinfo by default, and then
    this normalization can be skipped (https://github.com/apache/arrow/pull/49694).
    """
    if pytz is not None:
        # Convert any pytz timezones to zoneinfo / fixed offset timezones
        if any(
            isinstance(dtype, pd.DatetimeTZDtype)
            for dtype in df._mgr.get_unique_dtypes()
        ):
            col_indices = df._select_dtypes_indices(pd.DatetimeTZDtype)
            for i in col_indices:
                col = df.iloc[:, i]
                normalized_tz = _normalize_pytz_timezone(col.dtype.tz)
                if normalized_tz is not col.dtype.tz:
                    df.isetitem(i, col.dt.tz_convert(normalized_tz))

        df.index = _normalize_timezone_index(df.index)
        df.columns = _normalize_timezone_index(df.columns)

    return df
