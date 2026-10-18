# [MODULE DOC]
# Shared machinery for the text parsing engines. Public-ish contents: the
# ParserBase class (state and helpers common to the C, Python and pyarrow
# engine wrappers), a module-level defaults mapping of parser options (used by
# the readers module to detect non-default arguments and to fill in defaults),
# and free functions: date_converter (string column to datetimes),
# get_na_values (per-column missing-value sets), is_index_col (is an index
# column specification really set), validate_parse_dates_presence (check date
# columns exist) and evaluate_callable_usecols (resolve a callable column
# selector). The private function validating the column-selection argument is
# used by the base class constructor.
# Relationships: imported by the three engine wrapper modules and by the
# readers module; depends on the compiled parser module for missing-value
# constants and sanitizing helpers, on datetime tools, and on the arrow
# extension array for the pyarrow dtype backend.
# Import-time side effects: builds the defaults mapping only.
from __future__ import annotations

from collections import defaultdict
from copy import copy
import csv
from enum import Enum
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    parsers,
)
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    ParserError,
    ParserWarning,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_dict_like,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.missing import isna

from pandas import (
    DataFrame,
    DatetimeIndex,
    StringDtype,
)
from pandas.core import algorithms
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    BooleanArray,
    FloatingArray,
    IntegerArray,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
    ensure_index_from_sequences,
)
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools

from pandas.io.common import is_potential_multi_index

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeArg,
        Hashable,
        HashableT,
        Scalar,
        SequenceT,
    )


class ParserBase:
    # [CLASS DOC]
    # Base class for all parsing engine wrappers. Holds the parsed option state
    # shared by engines: column names (names, original names, column and index
    # names), index column spec, date parsing options (parse flag/list, date
    # parser, date format, day-first, cache flag), missing-value options,
    # dtype, converters, dtype backend, true/false values, header, validated
    # column selection and its inferred kind, bad-line policy, an implicit
    # index flag and a first-chunk flag used to special-case empty input.
    # Subclasses must supply their own reading logic; this class provides
    # index construction, type inference, date conversion and name cleaning
    # helpers, most marked final. Invariants: the usecols attribute is either a
    # set, a callable or None; header list-likes imply integer-only index
    # columns and no usecols/names.
    class BadLineHandleMethod(Enum):
        # [CLASS DOC]
        # Enumeration of policies for rows with too many fields: ERROR (0)
        # raise, WARN (1) warn and skip, SKIP (2) silently skip. The numeric
        # value is what the compiled C reader receives.
        ERROR = 0
        WARN = 1
        SKIP = 2

    _implicit_index: bool
    _first_chunk: bool
    keep_default_na: bool
    dayfirst: bool
    cache_dates: bool
    usecols_dtype: str | None

    def __init__(self, kwds) -> None:
        # [FUNCTION DOC]
        # Initializes shared parser state from the options mapping kwds.
        # Reads (without removing) names, index_col, na_values, na_fvalues,
        # na_filter (default false when absent), keep_default_na (default
        # true), dtype (shallow-copied), converters, dtype_backend, true_values,
        # false_values, header, usecols (required key), and on_bad_lines
        # (defaults to the error policy). Removes from the mapping (mutating
        # it) parse_dates, date_parser, date_format, dayfirst and cache_dates,
        # with defaults false, no-default sentinel, none, false and true.
        # Raises a type error if parse_dates is neither boolean/None nor a list
        # (None is coerced to false). When header is list-like (multi-row
        # header): raises a value error if usecols or names are given, and
        # normalizes index_col to a list of integers, raising a value error if
        # it contains non-integers. Sets all derived attributes (usecols and its
        # inferred kind via the private validator), the first-chunk flag to
        # true and the implicit-index flag to false.
        self._implicit_index = False

        self.names = kwds.get("names")
        self.orig_names: Sequence[Hashable] | None = None

        self.index_col = kwds.get("index_col", None)
        self.unnamed_cols: set = set()
        self.index_names: Sequence[Hashable] | None = None
        self.col_names: Sequence[Hashable] | None = None

        parse_dates = kwds.pop("parse_dates", False)
        if parse_dates is None or lib.is_bool(parse_dates):
            parse_dates = bool(parse_dates)
        elif not isinstance(parse_dates, list):
            raise TypeError(
                "Only booleans and lists are accepted for the 'parse_dates' parameter"
            )
        self.parse_dates: bool | list = parse_dates
        self.date_parser = kwds.pop("date_parser", lib.no_default)
        self.date_format = kwds.pop("date_format", None)
        self.dayfirst = kwds.pop("dayfirst", False)

        self.na_values = kwds.get("na_values")
        self.na_fvalues = kwds.get("na_fvalues")
        self.na_filter = kwds.get("na_filter", False)
        self.keep_default_na = kwds.get("keep_default_na", True)

        self.dtype = copy(kwds.get("dtype", None))
        self.converters = kwds.get("converters")
        self.dtype_backend = kwds.get("dtype_backend")

        self.true_values = kwds.get("true_values")
        self.false_values = kwds.get("false_values")
        self.cache_dates = kwds.pop("cache_dates", True)

        # validate header options for mi
        self.header = kwds.get("header")
        if is_list_like(self.header, allow_sets=False):
            if kwds.get("usecols"):
                raise ValueError(
                    "cannot specify usecols when specifying a multi-index header"
                )
            if kwds.get("names"):
                raise ValueError(
                    "cannot specify names when specifying a multi-index header"
                )

            # validate index_col that only contains integers
            if self.index_col is not None:
                # In this case we can pin down index_col as list[int]
                if is_integer(self.index_col):
                    self.index_col = [self.index_col]
                elif not (
                    is_list_like(self.index_col, allow_sets=False)
                    and all(map(is_integer, self.index_col))
                ):
                    raise ValueError(
                        "index_col must only contain integers of column positions "
                        "when specifying a multi-index header"
                    )
                else:
                    self.index_col = list(self.index_col)

        self._first_chunk = True

        self.usecols, self.usecols_dtype = _validate_usecols_arg(kwds["usecols"])

        # Fallback to error to pass a sketchy test(test_override_set_noconvert_columns)
        # Normally, this arg would get pre-processed earlier on
        self.on_bad_lines = kwds.get("on_bad_lines", self.BadLineHandleMethod.ERROR)

    def close(self) -> None:
        # [FUNCTION DOC]
        # Releases resources held by the parser. The base implementation does
        # nothing; engine wrappers override it.
        pass

    @final
    def _should_parse_dates(self, i: int) -> bool:
        # [FUNCTION DOC]
        # HELPER: decides whether the index column at position i should be
        # converted to dates. If the parse-dates option is a plain boolean,
        # returns it. Otherwise (a list), looks up the index name at position i
        # (if index names are known) and the original column position (the
        # position itself, or the index column entry at that position when an
        # index column spec exists) and returns true if either the position or
        # the name appears in the list.
        if isinstance(self.parse_dates, bool):
            return self.parse_dates
        else:
            if self.index_names is not None:
                name = self.index_names[i]
            else:
                name = None
            j = i if self.index_col is None else self.index_col[i]

            return (j in self.parse_dates) or (
                name is not None and name in self.parse_dates
            )

    @final
    def _extract_multi_indexer_columns(
        self,
        header,
        index_names: Sequence[Hashable] | None,
        passed_names: bool = False,
    ) -> tuple[
        Sequence[Hashable], Sequence[Hashable] | None, Sequence[Hashable] | None, bool
    ]:
        # [FUNCTION DOC]
        # HELPER: derives column labels when the header spans several rows.
        # Inputs: header is the list of header rows (each a list of labels);
        # index_names the caller's index names; passed_names whether the user
        # supplied names. Returns a four-tuple: names, index names, column-level
        # names and a passed-names flag.
        # With a single header row, returns that row, the unchanged index names,
        # no column-level names and the unchanged flag. Otherwise: removes the
        # last header row (mutating the header) and uses it, cleaned of
        # placeholder names, as the index names (the index_names argument is
        # thus ignored in this case); requires all remaining rows to have equal
        # length (raises a parser error otherwise); builds one tuple per
        # non-index column by zipping the rows, then reinserts the index
        # column positions as placeholders at their numeric positions; builds
        # column-level names from each row's entry at the first index column
        # (none if absent or an unnamed placeholder, or all none when there is
        # no index column). The returned flag is always true for multi-row
        # headers.
        """
        Extract and return the names, index_names, col_names if the column
        names are a MultiIndex.

        Parameters
        ----------
        header: list of lists
            The header rows
        index_names: list, optional
            The names of the future index
        passed_names: bool, default False
            A flag specifying if names where passed

        """
        if len(header) < 2:
            return header[0], index_names, None, passed_names

        # the names are the tuples of the header that are not the index cols
        # 0 is the name of the index, assuming index_col is a list of column
        # numbers
        ic = self.index_col
        if ic is None:
            ic = []

        if not isinstance(ic, (list, tuple, np.ndarray)):
            ic = [ic]
        sic = set(ic)

        # clean the index_names
        index_names = header.pop(-1)
        index_names, _, _ = self._clean_index_names(index_names, self.index_col)

        # extract the columns
        field_count = len(header[0])

        # check if header lengths are equal
        if not all(len(header_iter) == field_count for header_iter in header[1:]):
            raise ParserError("Header rows must have an equal number of columns.")

        def extract(r):
            return tuple(r[i] for i in range(field_count) if i not in sic)

        columns = list(zip(*(extract(r) for r in header), strict=True))
        names = columns.copy()
        for single_ic in sorted(ic):
            names.insert(single_ic, single_ic)

        # Clean the column names (if we have an index_col).
        if ic:
            col_names = [
                r[ic[0]]
                if ((r[ic[0]] is not None) and r[ic[0]] not in self.unnamed_cols)
                else None
                for r in header
            ]
        else:
            col_names = [None] * len(header)

        passed_names = True

        return names, index_names, col_names, passed_names

    @final
    def _maybe_make_multi_index_columns(
        self,
        columns: SequenceT,
        col_names: Sequence[Hashable] | None = None,
    ) -> SequenceT | MultiIndex:
        # [FUNCTION DOC]
        # HELPER: if the column labels are a non-empty sequence of tuples,
        # returns a multi-level index built from them, using col_names as level
        # names; otherwise returns the columns unchanged.
        # possibly create a column mi here
        if is_potential_multi_index(columns):
            columns_mi = cast("Sequence[tuple[Hashable, ...]]", columns)
            return MultiIndex.from_tuples(columns_mi, names=col_names)
        return columns

    @final
    def _make_index(
        self, alldata, columns, indexnamerow: list[Scalar] | None = None
    ) -> tuple[Index | None, Sequence[Hashable] | MultiIndex]:
        # [FUNCTION DOC]
        # HELPER: splits index columns out of parsed column data.
        # Inputs: alldata the list of column arrays (mutated: index columns are
        # removed); columns the column labels (mutated: index labels removed
        # unless an implicit index is in use); indexnamerow optional list of
        # index names read from a separate row.
        # Returns (index or None, columns possibly turned into a multi-level
        # index). When the index column option is a non-empty list, picks those
        # positions out of the data (string entries raise a value error as
        # invalid), removes them from the largest position down, aggregates
        # them into an index via the aggregation helper and, if an index name
        # row is given, names the index with the leading entries of it (as many
        # as the surplus over the column count). Otherwise no index is built.
        index: Index | None
        if isinstance(self.index_col, list) and len(self.index_col):
            to_remove = []
            indexes = []
            for idx in self.index_col:
                if isinstance(idx, str):
                    raise ValueError(f"Index {idx} invalid")
                to_remove.append(idx)
                indexes.append(alldata[idx])
            # remove index items from content and columns, don't pop in
            # loop
            for i in sorted(to_remove, reverse=True):
                alldata.pop(i)
                if not self._implicit_index:
                    columns.pop(i)
            index = self._agg_index(indexes)

            # add names for the index
            if indexnamerow:
                coffset = len(indexnamerow) - len(columns)
                index = index.set_names(indexnamerow[:coffset])
        else:
            index = None

        # maybe create a mi on the columns
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)

        return index, columns

    @final
    def _clean_mapping(self, mapping):
        # [FUNCTION DOC]
        # HELPER: converts integer keys of a per-column mapping (converters or
        # dtypes) to column names using the original names. Non-mapping input is
        # returned as is. Integer keys that are already names are kept. For a
        # default-valued mapping, the remaining original columns are given the
        # mapping's default via lookup so the result is explicit. Returns a new
        # plain dict. Requires original names to be set.
        """converts col numbers to names"""
        if not isinstance(mapping, dict):
            return mapping
        clean = {}
        # for mypy
        assert self.orig_names is not None

        for col, v in mapping.items():
            if isinstance(col, int) and col not in self.orig_names:
                col = self.orig_names[col]
            clean[col] = v
        if isinstance(mapping, defaultdict):
            remaining_cols = set(self.orig_names) - set(clean.keys())
            clean.update({col: mapping[col] for col in remaining_cols})
        return clean

    @final
    def _agg_index(self, index) -> Index:
        # [FUNCTION DOC]
        # HELPER: builds the final index from raw index column arrays.
        # Input is a sequence of arrays, one per index level. For each level:
        # optionally converts to dates when date parsing applies; picks the
        # missing-value set (empty when missing-value filtering is off; the
        # per-column sets when the missing-value option is a mapping and the
        # level has a name, else empty); looks up a per-level target dtype and
        # whether a converter exists (by index name); skips numeric/boolean
        # inference when the target is a string dtype or a converter exists;
        # infers types; and creates an index, with the cast dtype if given
        # (avoiding range-index inference) or via the generic sequence
        # constructor. Returns the single index or a multi-level index when
        # there are several levels. Raises a value error if index names and
        # arrays differ in count (strict pairing) when names are known.
        arrays = []
        converters = self._clean_mapping(self.converters)
        clean_dtypes = self._clean_mapping(self.dtype)

        if self.index_names is not None:
            names: Iterable = self.index_names
            zip_strict = True
        else:
            names = itertools.cycle([None])
            zip_strict = False
        for i, (arr, name) in enumerate(zip(index, names, strict=zip_strict)):
            if self._should_parse_dates(i):
                arr = date_converter(
                    arr,
                    col=self.index_names[i] if self.index_names is not None else None,
                    dayfirst=self.dayfirst,
                    cache_dates=self.cache_dates,
                    date_format=self.date_format,
                )

            if self.na_filter:
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                col_na_values = set()
                col_na_fvalues = set()

            if isinstance(self.na_values, dict):
                assert self.index_names is not None
                col_name = self.index_names[i]
                if col_name is not None:
                    col_na_values, col_na_fvalues = get_na_values(
                        col_name, self.na_values, self.na_fvalues, self.keep_default_na
                    )
                else:
                    col_na_values, col_na_fvalues = set(), set()

            cast_type = None
            index_converter = False
            if self.index_names is not None:
                if isinstance(clean_dtypes, dict):
                    cast_type = clean_dtypes.get(self.index_names[i], None)

                if isinstance(converters, dict):
                    index_converter = converters.get(self.index_names[i]) is not None

            try_num_bool = not (
                (cast_type and is_string_dtype(cast_type)) or index_converter
            )

            arr, _ = self._infer_types(
                arr, col_na_values | col_na_fvalues, cast_type is None, try_num_bool
            )
            if cast_type is not None:
                # Don't perform RangeIndex inference
                idx = Index(arr, name=name, dtype=cast_type, copy=False)
            else:
                idx = ensure_index_from_sequences([arr], [name])
            arrays.append(idx)

        if len(arrays) == 1:
            return arrays[0]
        else:
            return MultiIndex.from_arrays(arrays)

    @final
    def _set_noconvert_dtype_columns(
        self, col_indices: list[int], names: Sequence[Hashable]
    ) -> set[int]:
        # [FUNCTION DOC]
        # HELPER: computes positions of columns exempt from dtype conversion
        # (those used for date parsing). Does not itself mark anything; returns
        # a set of integer positions.
        # Inputs: col_indices positions corresponding to names; names column
        # labels. Builds a position lookup: for integer-valued usecols, the
        # sorted selected positions; for callable or string usecols, the given
        # col_indices; otherwise none. Each date-spec entry (when parse-dates
        # is a list, first validated for presence, raising a value error for
        # missing names) is mapped to a position: integers are translated via
        # the lookup when present, names are translated via their index in names
        # then col_indices. If parse-dates is a true flag instead, the index
        # column(s) are exempted.
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
        usecols: list[int] | list[str] | None
        noconvert_columns = set()
        if self.usecols_dtype == "integer":
            # A set of integers will be converted to a list in
            # the correct order every single time.
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ("empty", None):
            # The names attribute should have the correct columns
            # in the proper order for indexing with parse_dates.
            usecols = col_indices
        else:
            # Usecols is empty.
            usecols = None

        def _set(x) -> int:
            # [FUNCTION DOC]
            # HELPER: nested function converting a date specification entry (an
            # integer position relative to the selected columns, or a name) into
            # an absolute column position, using the enclosing lookup lists.
            # Integers are looked up in the selected-position list when one
            # exists; a non-integer result is treated as a name and translated
            # via its index in names.
            if usecols is not None and is_integer(x):
                x = usecols[x]

            if not is_integer(x):
                x = col_indices[names.index(x)]

            return x

        if isinstance(self.parse_dates, list):
            validate_parse_dates_presence(self.parse_dates, names)
            for val in self.parse_dates:
                noconvert_columns.add(_set(val))

        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    noconvert_columns.add(_set(k))
            elif self.index_col is not None:
                noconvert_columns.add(_set(self.index_col))

        return noconvert_columns

    @final
    def _infer_types(
        self, values, na_values, no_dtype_specified, try_num_bool: bool = True
    ) -> tuple[ArrayLike, int]:
        # [FUNCTION DOC]
        # HELPER: infers and converts the type of one parsed column.
        # Inputs: values an array of raw values; na_values the set of strings
        # and floats considered missing; no_dtype_specified true when the user
        # gave no explicit dtype for this column (enables nullable/arrow
        # outputs for non-default dtype backends); try_num_bool whether to try
        # numeric then boolean conversion (default true). Returns (converted
        # array, count of missing values found).
        # Numeric or boolean input: marks entries matching the non-string
        # missing values as NaN in place (integers are first promoted to float)
        # and returns. Object input with conversion enabled: tries numeric
        # conversion; on a value or type error falls back to sanitizing object
        # values (turning missing strings to NaN) and keeping them. With a
        # nullable backend, results become masked integer, boolean or float
        # arrays (an all-missing column becomes an integer array). If the
        # result is still object (and not leading with an integer), tries
        # boolean conversion using true/false values; nullable backends turn
        # booleans into a masked boolean array and remaining non-datetime
        # objects into a string-dtype array. Finally, for the pyarrow backend,
        # wraps results into an arrow extension array (an all-missing masked
        # result becomes an arrow null array). Requires pyarrow in that case.
        # KNOWN ISSUE: the docstring says try_num_bool defaults to try; the
        # default is the boolean true.
        """
        Infer types of values, possibly casting

        Parameters
        ----------
        values : ndarray
        na_values : set
        no_dtype_specified: Specifies if we want to cast explicitly
        try_num_bool : bool, default try
           try to cast values to numeric (first preference) or boolean

        Returns
        -------
        converted : ndarray or ExtensionArray
        na_count : int
        """
        na_count = 0
        if issubclass(values.dtype.type, (np.number, np.bool_)):
            # If our array has numeric dtype, we don't have to check for strings in isin
            na_values = np.array([val for val in na_values if not isinstance(val, str)])
            mask = algorithms.isin(values, na_values)
            na_count = mask.astype("uint8", copy=False).sum()
            if na_count > 0:
                if is_integer_dtype(values):
                    values = values.astype(np.float64)
                np.putmask(values, mask, np.nan)
            return values, na_count

        dtype_backend = self.dtype_backend
        non_default_dtype_backend = (
            no_dtype_specified and dtype_backend is not lib.no_default
        )
        result: ArrayLike

        if try_num_bool and is_object_dtype(values.dtype):
            # exclude e.g DatetimeIndex here
            try:
                result, result_mask = lib.maybe_convert_numeric(
                    values,
                    na_values,
                    False,
                    convert_to_masked_nullable=non_default_dtype_backend,  # type: ignore[arg-type]
                )
            except (ValueError, TypeError):
                # e.g. encountering datetime string gets ValueError
                #  TypeError can be raised in floatify
                na_count = parsers.sanitize_objects(values, na_values)
                result = values
            else:
                if non_default_dtype_backend:
                    if result_mask is None:
                        result_mask = np.zeros(result.shape, dtype=np.bool_)

                    if result_mask.all():
                        result = IntegerArray(
                            np.ones(result_mask.shape, dtype=np.int64), result_mask
                        )
                    elif is_integer_dtype(result):
                        result = IntegerArray(result, result_mask)
                    elif is_bool_dtype(result):
                        result = BooleanArray(result, result_mask)
                    elif is_float_dtype(result):
                        result = FloatingArray(result, result_mask)

                    na_count = result_mask.sum()
                else:
                    na_count = isna(result).sum()
        else:
            result = values
            if values.dtype == np.object_:
                na_count = parsers.sanitize_objects(values, na_values)

        if (
            result.dtype == np.object_
            and try_num_bool
            and (len(result) == 0 or not isinstance(result[0], int))
        ):
            result, bool_mask = libops.maybe_convert_bool(
                np.asarray(values),
                true_values=self.true_values,
                false_values=self.false_values,
                convert_to_masked_nullable=non_default_dtype_backend,  # type: ignore[arg-type]
            )
            if result.dtype == np.bool_ and non_default_dtype_backend:
                if bool_mask is None:
                    bool_mask = np.zeros(result.shape, dtype=np.bool_)
                result = BooleanArray(result, bool_mask)
            elif result.dtype == np.object_ and non_default_dtype_backend:
                # read_excel sends array of datetime objects
                if not lib.is_datetime_array(result, skipna=True):
                    dtype = StringDtype()
                    cls = dtype.construct_array_type()
                    result = cls._from_sequence(values, dtype=dtype)

        if dtype_backend == "pyarrow":
            pa = import_optional_dependency("pyarrow")
            if isinstance(result, np.ndarray):
                result = ArrowExtensionArray(pa.array(result, from_pandas=True))
            elif isinstance(result, BaseMaskedArray):
                if result._mask.all():
                    # We want an arrow null array here
                    result = ArrowExtensionArray(pa.array([None] * len(result)))
                else:
                    result = ArrowExtensionArray(
                        pa.array(result._data, mask=result._mask)
                    )
            else:
                result = ArrowExtensionArray(
                    pa.array(result.to_numpy(), from_pandas=True)
                )

        return result, na_count

    @overload
    def _do_date_conversions(
        self,
        names: Index,
        data: DataFrame,
    ) -> DataFrame:
        # [FUNCTION DOC]
        # Overload stub: frame input yields a frame. See the implementation.
        ...

    @overload
    def _do_date_conversions(
        self,
        names: Sequence[Hashable],
        data: Mapping[Hashable, ArrayLike],
    ) -> Mapping[Hashable, ArrayLike]:
        # [FUNCTION DOC]
        # Overload stub: mapping input yields a mapping. See the implementation.
        ...

    @final
    def _do_date_conversions(
        self,
        names: Sequence[Hashable] | Index,
        data: Mapping[Hashable, ArrayLike] | DataFrame,
    ) -> Mapping[Hashable, ArrayLike] | DataFrame:
        # [FUNCTION DOC]
        # HELPER: applies date conversion to the columns named in the
        # parse-dates option. Inputs: names column labels (used to resolve
        # integer specs that are not themselves keys of the data); data a
        # mapping of arrays or a frame, MUTATED in place (converted columns are
        # replaced). If parse-dates is not a list, returns data untouched. For
        # each entry, skips columns that are index columns or index names
        # (those are converted when the index is built); otherwise converts via
        # the module date converter with the parser's day-first, cache and
        # format options. Returns the same data object.
        if not isinstance(self.parse_dates, list):
            return data
        for colspec in self.parse_dates:
            if isinstance(colspec, int) and colspec not in data:
                colspec = names[colspec]
            if (isinstance(self.index_col, list) and colspec in self.index_col) or (
                isinstance(self.index_names, list) and colspec in self.index_names
            ):
                continue
            result = date_converter(
                data[colspec],
                col=colspec,
                dayfirst=self.dayfirst,
                cache_dates=self.cache_dates,
                date_format=self.date_format,
            )
            # error: Unsupported target for indexed assignment
            # ("Mapping[Hashable, ExtensionArray | ndarray[Any, Any]] | DataFrame")
            data[colspec] = result  # type: ignore[index]

        return data

    @final
    def _check_data_length(
        self,
        columns: Sequence[Hashable],
        data: Sequence[ArrayLike],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: warns when the number of data columns differs from the number
        # of column labels. Does nothing when an index column is set (truthy),
        # when lengths agree, or when there are no labels. Allows exactly one
        # extra trailing data column if it consists only of empty strings or
        # missing values (trailing delimiters). Otherwise emits a parser warning
        # about possible data loss. Returns nothing.
        # KNOWN ISSUE: the docstring claims a parser error is raised for an
        # index column that is not false; the code only warns (and skips the
        # check for any truthy index column).
        """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
        data: list of array-likes containing the data column-wise.
        """
        if not self.index_col and len(columns) != len(data) and columns:
            empty_str = is_object_dtype(data[-1]) and data[-1] == ""
            # error: No overload variant of "__ror__" of "ndarray" matches
            # argument type "ExtensionArray"
            empty_str_or_na = empty_str | isna(data[-1])  # type: ignore[operator]
            if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
                return
            warnings.warn(
                "Length of header or names does not match length of data. This leads "
                "to a loss of data with index_col=False.",
                ParserWarning,
                stacklevel=find_stack_level(),
            )

    @final
    def _validate_usecols_names(self, usecols: SequenceT, names: Sequence) -> SequenceT:
        # [FUNCTION DOC]
        # HELPER: verifies every selected column exists among names. Raises a
        # value error listing the missing ones; otherwise returns usecols
        # unchanged.
        """
        Validates that all usecols are present in a given
        list of names. If not, raise a ValueError that
        shows what usecols are missing.

        Parameters
        ----------
        usecols : iterable of usecols
            The columns to validate are present in names.
        names : iterable of names
            The column names to check against.

        Returns
        -------
        usecols : iterable of usecols
            The `usecols` parameter if the validation succeeds.

        Raises
        ------
        ValueError : Columns were missing. Error message will list them.
        """
        missing = [c for c in usecols if c not in names]
        if len(missing) > 0:
            raise ValueError(
                f"Usecols do not match columns, columns expected but not found: "
                f"{missing}"
            )

        return usecols

    @final
    def _clean_index_names(self, columns, index_col) -> tuple[list | None, list, list]:
        # [FUNCTION DOC]
        # HELPER: resolves index column specs into names and positions.
        # Inputs: columns the column labels; index_col the index spec. Returns
        # (index names, remaining column labels, index positions). If no index
        # is requested (none or false) returns no names and inputs unchanged.
        # Otherwise copies the inputs (no mutation of arguments): with empty
        # columns returns a list of none names per index column; for each spec,
        # a string is matched to the first equal label (its position replaces
        # the spec; if no label matches, the spec stays a string), an integer
        # is used as a position into the labels; the chosen labels are removed
        # from the returned columns. Names that are placeholder unnamed-column
        # markers are replaced by none.
        # KNOWN ISSUE: removal uses value equality, so duplicate labels may
        # remove an earlier occurrence than the indexed one.
        if not is_index_col(index_col):
            return None, columns, index_col

        columns = list(columns)

        # In case of no rows and multiindex columns we have to set index_names to
        # list of Nones GH#38292
        if not columns:
            return [None] * len(index_col), columns, index_col

        cp_cols = list(columns)
        index_names: list[str | int | None] = []

        # don't mutate
        index_col = list(index_col)

        for i, c in enumerate(index_col):
            if isinstance(c, str):
                index_names.append(c)
                for j, name in enumerate(cp_cols):
                    if name == c:
                        index_col[i] = j
                        columns.remove(name)
                        break
            else:
                name = cp_cols[c]
                columns.remove(name)
                index_names.append(name)

        # Only clean index names that were placeholders.
        for i, name in enumerate(index_names):
            if isinstance(name, str) and name in self.unnamed_cols:
                index_names[i] = None

        return index_names, columns, index_col

    @final
    def _get_empty_meta(
        self, columns: Sequence[HashableT], dtype: DtypeArg | None = None
    ) -> tuple[Index, list[HashableT], dict[HashableT, Series]]:
        # [FUNCTION DOC]
        # HELPER: builds the empty result structure when the input had no data
        # rows. Inputs: columns labels; dtype a single dtype, a mapping from
        # label or integer position to dtype, or none. Returns (empty index,
        # list of remaining column labels, mapping of label to an empty series
        # of the column's dtype).
        # Dtypes: a non-mapping dtype applies to all columns (none meaning
        # object); a mapping is keyed by label (integer keys translated via
        # positions) with unmapped columns getting none. Index: when index
        # column and index names are both set, creates an empty index (or
        # multi-level index) with those names and dtypes, then removes the index
        # columns from the labels. Otherwise a default empty index.
        # Side effect: sorts the stored index column list in place (mutating
        # parser state) before removing columns.
        columns = list(columns)

        index_col = self.index_col
        index_names = self.index_names

        # Convert `dtype` to a defaultdict of some kind.
        # This will enable us to write `dtype[col_name]`
        # without worrying about KeyError issues later on.
        dtype_dict: defaultdict[Hashable, Any]
        if not is_dict_like(dtype):
            # if dtype == None, default will be object.
            dtype_dict = defaultdict(lambda: dtype)
        else:
            dtype = cast(dict, dtype)
            dtype_dict = defaultdict(
                lambda: None,
                {columns[k] if is_integer(k) else k: v for k, v in dtype.items()},
            )

        # Even though we have no data, the "index" of the empty DataFrame
        # could for example still be an empty MultiIndex. Thus, we need to
        # check whether we have any index columns specified, via either:
        #
        # 1) index_col (column indices)
        # 2) index_names (column names)
        #
        # Both must be non-null to ensure a successful construction. Otherwise,
        # we have to create a generic empty Index.
        index: Index
        if (index_col is None or index_col is False) or index_names is None:
            index = default_index(0)
        else:
            # TODO: We could return default_index(0) if dtype_dict[name] is None
            data = [
                Index([], name=name, dtype=dtype_dict[name]) for name in index_names
            ]
            if len(data) == 1:
                index = data[0]
            else:
                index = MultiIndex.from_arrays(data)
            index_col.sort()

            for i, n in enumerate(index_col):
                columns.pop(n - i)

        col_dict = {
            col_name: Series([], dtype=dtype_dict[col_name]) for col_name in columns
        }

        return index, columns, col_dict


def date_converter(
    date_col,
    col: Hashable,
    dayfirst: bool = False,
    cache_dates: bool = True,
    date_format: dict[Hashable, str] | str | None = None,
):
    # [FUNCTION DOC]
    # Converts one column of values to datetimes.
    # Inputs: date_col an array; col the column label (used to pick a format
    # when date_format is a mapping); dayfirst whether ambiguous dates are
    # day-first; cache_dates whether to use the conversion cache; date_format a
    # single format string, a per-column mapping, or none (infer).
    # Returns the input unchanged if it is already datetime or timedelta
    # typed. Otherwise converts to strings and parses (timezone-naive
    # handling, no forced UTC); on a value or type error returns the string
    # array instead of raising. A parsed datetime index is returned as a
    # writable object/datetime array; other results return their underlying
    # values.
    if date_col.dtype.kind in "Mm":
        return date_col

    date_fmt = date_format.get(col) if isinstance(date_format, dict) else date_format

    str_objs = lib.ensure_string_array(np.asarray(date_col))
    try:
        result = tools.to_datetime(
            str_objs,
            format=date_fmt,
            utc=False,
            dayfirst=dayfirst,
            cache=cache_dates,
        )
    except (ValueError, TypeError):
        # test_usecols_with_parse_dates4
        # test_multi_index_parse_dates
        return str_objs

    if isinstance(result, DatetimeIndex):
        arr = result.to_numpy()
        arr.flags.writeable = True
        return arr
    return result._values


# Module-level mapping of default values for parser options (names as accepted
# by the readers). Used by the readers module to fill in defaults and detect
# options explicitly set by the caller. Includes the bad-line policy default of
# error and the unset dtype-backend sentinel.
parser_defaults = {
    "delimiter": None,
    "escapechar": None,
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "doublequote": True,
    "skipinitialspace": False,
    "lineterminator": None,
    "header": "infer",
    "index_col": None,
    "names": None,
    "skiprows": None,
    "skipfooter": 0,
    "nrows": None,
    "na_values": None,
    "keep_default_na": True,
    "true_values": None,
    "false_values": None,
    "converters": None,
    "dtype": None,
    "cache_dates": True,
    "thousands": None,
    "comment": None,
    "decimal": ".",
    # 'engine': 'c',
    "parse_dates": False,
    "dayfirst": False,
    "date_format": None,
    "usecols": None,
    # 'iterator': False,
    "chunksize": None,
    "encoding": None,
    "compression": None,
    "skip_blank_lines": True,
    "encoding_errors": "strict",
    "on_bad_lines": ParserBase.BadLineHandleMethod.ERROR,
    "dtype_backend": lib.no_default,
}


def get_na_values(col, na_values, na_fvalues, keep_default_na: bool):
    # [FUNCTION DOC]
    # Resolves the missing-value sets for one column. If na_values is a
    # mapping: returns the string and float sets stored for the column when it
    # is a key (the float mapping must also contain the key); else the default
    # string set and an empty set when keep_default_na is true, or two empty
    # sets otherwise. If na_values is not a mapping, returns both inputs
    # unchanged. Returns a pair (string-valued set, float-valued set).
    """
    Get the NaN values for a given column.

    Parameters
    ----------
    col : str
        The name of the column.
    na_values : array-like, dict
        The object listing the NaN values as strings.
    na_fvalues : array-like, dict
        The object listing the NaN values as floats.
    keep_default_na : bool
        If `na_values` is a dict, and the column is not mapped in the
        dictionary, whether to return the default NaN values or the empty set.

    Returns
    -------
    nan_tuple : A length-two tuple composed of

        1) na_values : the string NaN values for that column.
        2) na_fvalues : the float NaN values for that column.
    """
    if isinstance(na_values, dict):
        if col in na_values:
            return na_values[col], na_fvalues[col]
        else:
            if keep_default_na:
                return STR_NA_VALUES, set()

            return set(), set()
    else:
        return na_values, na_fvalues


def is_index_col(col) -> bool:
    # [FUNCTION DOC]
    # Returns true when the index column option designates real columns, that
    # is, it is neither none nor the boolean false (zero and empty lists count
    # as set).
    return col is not None and col is not False


def validate_parse_dates_presence(
    parse_dates: bool | list, columns: Sequence[Hashable]
) -> set:
    # [FUNCTION DOC]
    # Checks that the columns requested for date parsing exist. Inputs:
    # parse_dates the option (only a list is examined; anything else returns an
    # empty set); columns the available labels. Names that are strings must be
    # present; non-string entries that are labels are accepted as is; other
    # entries are treated as positions and resolved to labels by indexing.
    # Raises a value error listing the sorted missing string names. Returns the
    # set of resolved labels (excluding missing ones).
    # KNOWN ISSUE: the docstring omits the parse_dates parameter and describes
    # return as names to be parsed; an out-of-range position raises an index
    # error rather than a value error.
    """
    Check if parse_dates are in columns.

    If user has provided names for parse_dates, check if those columns
    are available.

    Parameters
    ----------
    columns : list
        List of names of the dataframe.

    Returns
    -------
    The names of the columns which will get parsed later if a list
    is given as specification.

    Raises
    ------
    ValueError
        If column to parse_date is not in dataframe.

    """
    if not isinstance(parse_dates, list):
        return set()

    missing = set()
    unique_cols = set()
    for col in parse_dates:
        if isinstance(col, str):
            if col not in columns:
                missing.add(col)
            else:
                unique_cols.add(col)
        elif col in columns:
            unique_cols.add(col)
        else:
            unique_cols.add(columns[col])
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing column provided to 'parse_dates': '{missing_cols}'")
    return unique_cols


def _validate_usecols_arg(usecols):
    # [FUNCTION DOC]
    # HELPER: validates the column selection option used by the base
    # constructor. None passes through as (none, none); a callable passes
    # through as (callable, none). Non-list-like values (including strings)
    # and list-likes whose inferred kind is not empty, integer or string raise
    # a value error. Valid list-likes are converted to a set and returned with
    # the inferred kind (empty, integer or string).
    """
    Validate the 'usecols' parameter.

    Checks whether or not the 'usecols' parameter contains all integers
    (column selection by index), strings (column by name) or is a callable.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    usecols : list-like, callable, or None
        List of columns to use when parsing or a callable that can be used
        to filter a list of table columns.

    Returns
    -------
    usecols_tuple : tuple
        A tuple of (verified_usecols, usecols_dtype).

        'verified_usecols' is either a set if an array-like is passed in or
        'usecols' if a callable or None is passed in.

        'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
        is passed in or None if a callable or None is passed in.
    """
    msg = (
        "'usecols' must either be list-like of all strings, all unicode, "
        "all integers or a callable."
    )
    if usecols is not None:
        if callable(usecols):
            return usecols, None

        if not is_list_like(usecols):
            # see gh-20529
            #
            # Ensure it is iterable container but not string.
            raise ValueError(msg)

        usecols_dtype = lib.infer_dtype(usecols, skipna=False)

        if usecols_dtype not in ("empty", "integer", "string"):
            raise ValueError(msg)

        usecols = set(usecols)

        return usecols, usecols_dtype
    return usecols, None


@overload
def evaluate_callable_usecols(
    usecols: Callable[[Hashable], object],
    names: Iterable[Hashable],
) -> set[int]:
    # [FUNCTION DOC]
    # Overload stub: a callable selector yields a set of positions. See the
    # implementation.
    ...


@overload
def evaluate_callable_usecols(
    usecols: SequenceT, names: Iterable[Hashable]
) -> SequenceT:
    # [FUNCTION DOC]
    # Overload stub: a non-callable selector is returned unchanged. See the
    # implementation.
    ...


def evaluate_callable_usecols(
    usecols: Callable[[Hashable], object] | SequenceT,
    names: Iterable[Hashable],
) -> SequenceT | set[int]:
    # [FUNCTION DOC]
    # Resolves a column selector against column labels. If usecols is callable,
    # evaluates it on each label and returns the set of positions for which it
    # is truthy; otherwise returns usecols unchanged (including none).
    """
    Check whether or not the 'usecols' parameter
    is a callable.  If so, enumerates the 'names'
    parameter and returns a set of indices for
    each entry in 'names' that evaluates to True.
    If not a callable, returns 'usecols'.
    """
    if callable(usecols):
        return {i for i, name in enumerate(names) if usecols(name)}
    return usecols
