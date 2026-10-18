# [MODULE DOC]
# Pure-Python parsing engine for delimited and fixed-width text. Public
# content: PythonParser (the delimited engine, a parser base subclass used by
# the readers module for the python engine and by the Excel reader for nested
# lists), FixedWidthReader (an iterator that slices lines into fields by
# column ranges, optionally inferring the ranges from sample lines) and
# FixedWidthFieldParser (a parser specialization using that reader). A private
# validator checks the skipfooter argument and a module constant holds the
# byte-order-mark character that is stripped from the start of the data.
# Relationships: builds on the shared parser base for index, date, type
# inference and name handling; uses the standard csv module for tokenizing.
# No import-time side effects beyond imports.
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import csv
from io import StringIO
import re
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Literal,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._typing import Scalar
from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_extension_array_dtype,
    is_integer,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.inference import is_dict_like

from pandas.core import algorithms
from pandas.core.arrays import (
    Categorical,
    ExtensionArray,
)
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import Index

from pandas.io.common import (
    dedup_names,
    is_potential_multi_index,
)
from pandas.io.parsers.base_parser import (
    ParserBase,
    evaluate_callable_usecols,
    get_na_values,
    parser_defaults,
    validate_parse_dates_presence,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        ReadCsvBuffer,
        T,
    )

    from pandas import (
        MultiIndex,
        Series,
    )

# BOM character (byte order mark)
# This exists at the beginning of a file to indicate endianness
# of a file (stream). Unfortunately, this marker screws up parsing,
# so we need to remove it if we see it.
_BOM = "\ufeff"


class PythonParser(ParserBase):
    # [CLASS DOC]
    # Delimited-text parsing engine written in Python. Extends the shared
    # parser base. State: data (an iterator of string rows or a nested list for
    # Excel input), buf (lines read ahead and pushed back), pos (rows consumed
    # including skipped ones), line_pos (rows consumed that were not skipped),
    # dialect-related settings (delimiter, quote and escape characters, quoting,
    # doublequote, initial-space skipping, line terminator, blank-line
    # skipping, comment character, thousands and decimal markers), skip
    # settings (a skip-rows spec and a predicate derived from it, skip-footer
    # count), column bookkeeping (columns, original names, original column
    # count, selected column positions, columns excluded from thousands
    # handling) and an Excel-only flag for index-name rows. Main methods: read
    # (return a chunk), the line fetching/cleaning helpers, and the header and
    # index inference helpers. Subclassed for fixed-width input.
    _no_thousands_columns: set[int]

    def __init__(self, f: ReadCsvBuffer[str] | list, **kwds) -> None:
        # [FUNCTION DOC]
        # Sets up the parser, reads the header and infers columns and index.
        # Inputs: f a text stream with a line-reading method, or a nested list
        # of rows (Excel); kwds the option mapping, which must contain skiprows,
        # skipfooter, delimiter, quotechar, escapechar, doublequote,
        # skipinitialspace, lineterminator, quoting, skip_blank_lines,
        # thousands, decimal and comment (plus the base class keys); the Excel
        # flag has_index_names is optional (default false).
        # Behavior: runs base initialization; builds the skip predicate (a
        # callable skiprows is used directly, otherwise membership in the
        # collection); validates skipfooter (value error if not a non-negative
        # integer); creates the row source (a list is used as is, anything else
        # must have a line-reading method and is wrapped by the reader factory);
        # infers raw columns from the header, resolves multi-row header columns,
        # derives original names, index names and cleans them; selects default
        # column positions when none were chosen; computes columns exempt from
        # thousands handling. Raises a value error unless the decimal marker has
        # length one. May consume lines from the stream (header, and one or two
        # lookahead lines buffered for later).
        super().__init__(kwds)

        self.data: Iterator[list[str]] | list[list[Scalar]] = []
        self.buf: list = []
        self.pos = 0
        self.line_pos = 0

        self.skiprows = kwds["skiprows"]

        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            # (lambda: membership test of a row number in the skip-rows
            # collection; returns a boolean)
            self.skipfunc = lambda x: x in self.skiprows

        self.skipfooter = _validate_skipfooter_arg(kwds["skipfooter"])
        self.delimiter = kwds["delimiter"]

        self.quotechar = kwds["quotechar"]
        if isinstance(self.quotechar, str):
            self.quotechar = str(self.quotechar)

        self.escapechar = kwds["escapechar"]
        self.doublequote = kwds["doublequote"]
        self.skipinitialspace = kwds["skipinitialspace"]
        self.lineterminator = kwds["lineterminator"]
        self.quoting = kwds["quoting"]
        self.skip_blank_lines = kwds["skip_blank_lines"]

        # Passed from read_excel
        self.has_index_names = kwds.get("has_index_names", False)

        self.thousands = kwds["thousands"]
        self.decimal = kwds["decimal"]

        self.comment = kwds["comment"]

        # Set self.data to something that can read lines.
        if isinstance(f, list):
            # read_excel: f is a nested list, can contain non-str
            self.data = f
        else:
            assert hasattr(f, "readline")
            # yields list of str
            self.data = self._make_reader(f)

        # Get columns in two steps: infer from data, then
        # infer column indices from self.usecols if it is specified.
        self._col_indices: list[int] | None = None
        columns: list[list[Scalar | None]]
        (
            columns,
            self.num_original_columns,
            self.unnamed_cols,
        ) = self._infer_columns()

        # Now self.columns has the set of columns that we will process.
        # The original set is stored in self.original_columns.
        # error: Cannot determine type of 'index_names'
        (
            self.columns,
            self.index_names,
            self.col_names,
            _,
        ) = self._extract_multi_indexer_columns(
            columns,
            self.index_names,
        )

        # get popped off for index
        self.orig_names: list[Hashable] = list(self.columns)

        index_names, self.orig_names, self.columns = self._get_index_name()
        if self.index_names is None:
            self.index_names = index_names

        if self._col_indices is None:
            self._col_indices = list(range(len(self.columns)))

        self._no_thousands_columns = self._set_no_thousand_columns()

        if len(self.decimal) != 1:
            raise ValueError("Only length-1 decimal markers supported")

    @cache_readonly
    def num(self) -> re.Pattern:
        # [FUNCTION DOC]
        # Cached read-only property. Returns a compiled pattern matching strings
        # that look like numbers: optional sign, digits (with the thousands
        # marker allowed between digit groups when one is configured), an
        # optional decimal part using the configured decimal marker, and an
        # optional exponent. Used to decide whether a text cell is numeric
        # before removing thousands separators or replacing the decimal marker.
        decimal = re.escape(self.decimal)
        if self.thousands is None:
            regex = rf"^[\-\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\-?[0-9]+)?$"
        else:
            thousands = re.escape(self.thousands)
            regex = (
                rf"^[\-\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?"
                rf"([0-9]?(E|e)\-?[0-9]+)?$"
            )
        return re.compile(regex)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> Iterator[list[str]]:
        # [FUNCTION DOC]
        # HELPER: wraps the text stream into an iterator of field lists.
        # Input f is the stream. Three modes by delimiter: (a) a single-character
        # delimiter uses the standard csv reader in strict mode with a dialect
        # built from the parser settings (a custom line terminator raises a
        # value error); rows to skip are first discarded from the stream so the
        # csv reader never sees them; (b) no delimiter: reads lines until one
        # that is neither skipped nor a comment, lets the csv sniffer detect the
        # delimiter from it, pushes that parsed line into the buffer and
        # advances the position counters; (c) a longer delimiter is treated as a
        # regular expression, and each stripped line is split with it (the
        # first line is read separately, which makes the file-iteration
        # continue after it). Returns the iterator. May raise csv errors from
        # sniffing (e.g. when no delimiter can be determined).
        # KNOWN ISSUE: in the sniffing mode the loop reading the first
        # valid line never checks for end of input, so an empty or
        # all-skipped file ends with a sniffer error rather than an
        # empty-data error.
        sep = self.delimiter

        if sep is None or len(sep) == 1:
            if self.lineterminator:
                raise ValueError(
                    "Custom line terminators not supported in python parser (yet)"
                )

            class MyDialect(csv.Dialect):
                # [CLASS DOC]
                # HELPER: local csv dialect built from the enclosing parser's
                # settings (delimiter, quote and escape characters, doublequote,
                # initial-space skipping, quoting mode) with a newline line
                # terminator. The delimiter may be overridden after creation
                # (sniffed or explicit).
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = self.escapechar
                doublequote = self.doublequote
                skipinitialspace = self.skipinitialspace
                quoting = self.quoting
                lineterminator = "\n"

            dia = MyDialect

            if sep is not None:
                dia.delimiter = sep
                # Skip rows at file level before csv.reader sees them
                # prevents CSV parsing errors on lines that will be discarded
                if self.skiprows is not None:
                    while self.skipfunc(self.pos):
                        line = f.readline()
                        if not line:
                            break
                        self.pos += 1
            else:
                # attempt to sniff the delimiter from the first valid line,
                # i.e. no comment line and not in skiprows
                line = f.readline()
                lines = self._check_comments([[line]])[0]
                while self.skipfunc(self.pos) or not lines:
                    self.pos += 1
                    line = f.readline()
                    lines = self._check_comments([[line]])[0]
                lines_str = cast(list[str], lines)

                # since `line` was a string, lines will be a list containing
                # only a single string
                line = lines_str[0]

                self.pos += 1
                self.line_pos += 1
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter

                # Note: encoding is irrelevant here
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))

            # Note: encoding is irrelevant here
            reader = csv.reader(f, dialect=dia, strict=True)

        else:

            def _read():
                # [FUNCTION DOC]
                # HELPER: generator for the regular-expression delimiter mode.
                # Compiles the delimiter once, yields the field list of the
                # first line (read explicitly) and then of every remaining line
                # of the stream, each stripped of surrounding whitespace before
                # splitting.
                line = f.readline()
                pat = re.compile(sep)

                yield pat.split(line.strip())

                for line in f:
                    yield pat.split(line.strip())

            reader = _read()

        return reader

    def read(
        self, rows: int | None = None
    ) -> tuple[
        Index | None,
        Sequence[Hashable] | MultiIndex,
        Mapping[Hashable, ArrayLike | Series],
    ]:
        # [FUNCTION DOC]
        # Reads up to rows data rows (all remaining when none) and returns
        # (index or none, column labels, mapping of label to converted column).
        # Fetches cleaned rows; if the source is exhausted, the first call
        # proceeds with no content (to yield an empty result with the right
        # metadata) while later calls close the parser and re-raise the
        # end-of-iteration signal. With no content, builds the empty result from
        # deduplicated original names. Otherwise: if the Excel index-names flag
        # is set and the first row consists solely of empty entries matching the
        # column count, that row is treated as the index name row and removed;
        # transposes rows into columns, splits off the implicit-index columns,
        # converts types per column, applies date conversions and builds the
        # index and final column labels.
        try:
            content = self._get_lines(rows)
        except StopIteration:
            if self._first_chunk:
                content = []
            else:
                self.close()
                raise

        # done with first read, next time raise StopIteration
        self._first_chunk = False

        index: Index | None
        columns: Sequence[Hashable] = list(self.orig_names)
        if not content:  # pragma: no cover
            # DataFrame with the right metadata, even though it's length 0
            # error: Cannot determine type of 'index_col'
            names = dedup_names(
                self.orig_names,
                is_potential_multi_index(
                    self.orig_names,
                    self.index_col,
                ),
            )
            index, columns, col_dict = self._get_empty_meta(
                names,
                self.dtype,
            )
            conv_columns = self._maybe_make_multi_index_columns(columns, self.col_names)
            return index, conv_columns, col_dict

        # handle new style for names in index
        indexnamerow = None
        if self.has_index_names and sum(
            int(v == "" or v is None) for v in content[0]
        ) == len(columns):
            indexnamerow = content[0]
            content = content[1:]

        alldata = self._rows_to_cols(content)
        data, columns = self._exclude_implicit_index(alldata)

        conv_data = self._convert_data(data)
        conv_data = self._do_date_conversions(columns, conv_data)

        index, result_columns = self._make_index(alldata, columns, indexnamerow)

        return index, result_columns, conv_data

    def _exclude_implicit_index(
        self,
        alldata: list[np.ndarray],
    ) -> tuple[Mapping[Hashable, np.ndarray], Sequence[Hashable]]:
        # [FUNCTION DOC]
        # HELPER: separates data columns from implicit index columns.
        # Input alldata is the list of all column arrays. Computes deduplicated
        # original names, offsets by the number of index columns when an
        # implicit index is in use, checks the data length against the names
        # (possibly warning), and returns (mapping of name to array for each name
        # that has data, the deduplicated names). Names beyond the available
        # data are omitted from the mapping.
        # error: Cannot determine type of 'index_col'
        names = dedup_names(
            self.orig_names,
            is_potential_multi_index(
                self.orig_names,
                self.index_col,
            ),
        )

        offset = 0
        if self._implicit_index:
            offset = len(self.index_col)

        len_alldata = len(alldata)
        self._check_data_length(names, alldata)

        return {
            name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata
        }, names

    # legacy
    def get_chunk(
        self, size: int | None = None
    ) -> tuple[
        Index | None,
        Sequence[Hashable] | MultiIndex,
        Mapping[Hashable, ArrayLike | Series],
    ]:
        # [FUNCTION DOC]
        # Legacy convenience wrapper around read. Reads size rows; when size is
        # none, uses the chunk size attribute. Returns what read returns.
        # KNOWN ISSUE: the chunk size attribute is not defined in this class
        # (a type-checker suppression hides this), so calling with no size fails
        # with an attribute error unless something else sets it.
        if size is None:
            # error: "PythonParser" has no attribute "chunksize"
            size = self.chunksize  # type: ignore[attr-defined]
        return self.read(rows=size)

    def _convert_data(
        self,
        data: Mapping[Hashable, np.ndarray],
    ) -> Mapping[Hashable, ArrayLike]:
        # [FUNCTION DOC]
        # HELPER: prepares per-column options and converts the raw columns.
        # Input data maps column name to raw object array. Resolves converters
        # and dtypes so that integer keys become names. Resolves missing-value
        # markers: when they are a mapping, integer keys are turned into names
        # (entries with a none key are dropped) for both string and float marker
        # mappings; otherwise they are used as is. Delegates to the conversion
        # routine and returns its mapping of name to converted array.
        # apply converters
        clean_conv = self._clean_mapping(self.converters)
        clean_dtypes = self._clean_mapping(self.dtype)

        # Apply NA values.
        clean_na_values = {}
        clean_na_fvalues = {}

        if isinstance(self.na_values, dict):
            for col in self.na_values:
                if col is not None:
                    na_value = self.na_values[col]
                    na_fvalue = self.na_fvalues[col]

                    if isinstance(col, int) and col not in self.orig_names:
                        col = self.orig_names[col]

                    clean_na_values[col] = na_value
                    clean_na_fvalues[col] = na_fvalue
        else:
            clean_na_values = self.na_values
            clean_na_fvalues = self.na_fvalues

        return self._convert_to_ndarrays(
            data,
            clean_na_values,
            clean_na_fvalues,
            clean_conv,
            clean_dtypes,
        )

    @final
    def _convert_to_ndarrays(
        self,
        dct: Mapping,
        na_values,
        na_fvalues,
        converters=None,
        dtypes=None,
    ) -> dict[Any, np.ndarray]:
        # [FUNCTION DOC]
        # HELPER: converts each raw column according to converters, dtype,
        # missing-value markers and the date list. Inputs: dct name to raw
        # array; na_values and na_fvalues string and float missing markers
        # (global or per-column); converters per-column functions (optional);
        # dtypes a mapping or a single dtype (optional). Returns a new dict of
        # name to converted array.
        # Per column: the missing set is empty unless missing-value filtering is
        # on. Date columns are left unconverted except that missing markers are
        # replaced with NaN in place so date parsing works. With a converter:
        # warns that the dtype is ignored if both given; applies the converter
        # to every value (on a value error, retries skipping entries matching
        # missing markers), then infers types without numeric/boolean
        # attempts. Without one: skips numeric/boolean attempts when the target
        # is a string or extension dtype; infers types; if a dtype is given and
        # differs from the inferred one (or is an extension type), raises a
        # value error when casting a column with missing values to boolean (for
        # non-extension types), else casts through the cast helper. Raises a
        # value error if date columns are not all present.
        # KNOWN ISSUE: in the converter retry path, the missing mask is built
        # from the global missing marker collection even when it is a
        # mapping, ignoring the per-column markers.
        result = {}
        parse_date_cols = validate_parse_dates_presence(self.parse_dates, self.columns)
        for c, values in dct.items():
            conv_f = None if converters is None else converters.get(c, None)
            if isinstance(dtypes, dict):
                cast_type = dtypes.get(c, None)
            else:
                # single dtype or None
                cast_type = dtypes

            if self.na_filter:
                col_na_values, col_na_fvalues = get_na_values(
                    c, na_values, na_fvalues, self.keep_default_na
                )
            else:
                col_na_values, col_na_fvalues = set(), set()

            if c in parse_date_cols:
                # GH#26203 Do not convert columns which get converted to dates
                # but replace nans to ensure to_datetime works
                mask = algorithms.isin(values, set(col_na_values) | col_na_fvalues)  # pyright: ignore[reportArgumentType]
                np.putmask(values, mask, np.nan)
                result[c] = values
                continue

            if conv_f is not None:
                # conv_f applied to data before inference
                if cast_type is not None:
                    warnings.warn(
                        (
                            "Both a converter and dtype were specified "
                            f"for column {c} - only the converter will be used."
                        ),
                        ParserWarning,
                        stacklevel=find_stack_level(),
                    )

                try:
                    values = lib.map_infer(values, conv_f)
                except ValueError:
                    mask = algorithms.isin(values, list(na_values)).view(np.uint8)
                    values = lib.map_infer_mask(values, conv_f, mask)

                cvals, na_count = self._infer_types(
                    values,
                    set(col_na_values) | col_na_fvalues,
                    cast_type is None,
                    try_num_bool=False,
                )
            else:
                is_ea = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype = is_ea or is_string_dtype(cast_type)
                # skip inference if specified dtype is object
                # or casting to an EA
                try_num_bool = not (cast_type and is_str_or_ea_dtype)

                # general type inference and conversion
                cvals, na_count = self._infer_types(
                    values,
                    set(col_na_values) | col_na_fvalues,
                    cast_type is None,
                    try_num_bool,
                )

                # type specified in dtype param or cast_type is an EA
                if cast_type is not None:
                    cast_type = pandas_dtype(cast_type)
                if cast_type and (cvals.dtype != cast_type or is_ea):
                    if not is_ea and na_count > 0:
                        if is_bool_dtype(cast_type):
                            raise ValueError(f"Bool column has NA values in column {c}")
                    cvals = self._cast_types(cvals, cast_type, c)

            result[c] = cvals
        return result

    @final
    def _cast_types(self, values: ArrayLike, cast_type: DtypeObj, column) -> ArrayLike:
        # [FUNCTION DOC]
        # HELPER: casts an inferred column to an explicit target dtype.
        # Inputs: values the array; cast_type the target dtype object; column
        # the label, used only in error text. Returns the cast array.
        # Categorical targets: non-object input without known categories is first
        # stringified (matching the C engine); categories are the unique
        # non-missing values and codes are looked up, honoring configured true
        # values. Other extension targets: built from string sequences by the
        # array type (boolean targets additionally receive true, false and
        # missing markers, with values stringified first); a missing
        # implementation raises a not-implemented error naming the array type.
        # Extension-array input with a numpy target: element-wise cast without
        # copying. String targets: converted to string objects keeping missing
        # as is. Anything else: a copying array cast, with a value error naming
        # the column and dtype on failure.
        # KNOWN ISSUE: a source comment notes the string conversion skips
        # missing values here while the categorical path does not; the
        # difference is untested.
        """
        Cast values to specified type

        Parameters
        ----------
        values : ndarray or ExtensionArray
        cast_type : np.dtype or ExtensionDtype
           dtype to cast values to
        column : string
            column name - used only for error reporting

        Returns
        -------
        converted : ndarray or ExtensionArray
        """
        if isinstance(cast_type, CategoricalDtype):
            known_cats = cast_type.categories is not None

            if not is_object_dtype(values.dtype) and not known_cats:
                # TODO: this is for consistency with
                # c-parser which parses all categories
                # as strings
                values = lib.ensure_string_array(
                    values, skipna=False, convert_na_value=False
                )

            cats = Index(values, copy=False).unique().dropna()
            values = Categorical._from_inferred_categories(
                cats, cats.get_indexer(values), cast_type, true_values=self.true_values
            )

        # use the EA's implementation of casting
        elif isinstance(cast_type, ExtensionDtype):
            array_type = cast_type.construct_array_type()
            try:
                if isinstance(cast_type, BooleanDtype):
                    # error: Unexpected keyword argument "true_values" for
                    # "_from_sequence_of_strings" of "ExtensionArray"
                    values_str = [str(val) for val in values]
                    return array_type._from_sequence_of_strings(  # type: ignore[call-arg]
                        values_str,
                        dtype=cast_type,
                        true_values=self.true_values,  # pyright: ignore[reportCallIssue]
                        false_values=self.false_values,  # pyright: ignore[reportCallIssue]
                        none_values=self.na_values,  # pyright: ignore[reportCallIssue]
                    )
                else:
                    return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"Extension Array: {array_type} must implement "
                    "_from_sequence_of_strings in order to be used in parser methods"
                ) from err

        elif isinstance(values, ExtensionArray):
            values = values.astype(cast_type, copy=False)
        elif issubclass(cast_type.type, str):
            # TODO: why skipna=True here and False above? some tests depend
            #  on it here, but nothing fails if we change it above
            #  (as no tests get there as of 2022-12-06)
            values = lib.ensure_string_array(
                values, skipna=True, convert_na_value=False
            )
        else:
            try:
                values = astype_array(values, cast_type, copy=True)
            except ValueError as err:
                raise ValueError(
                    f"Unable to convert column {column} to type {cast_type}"
                ) from err
        return values

    @cache_readonly
    def _have_mi_columns(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: cached read-only property; true when the header option is a
        # list, tuple or array with more than one entry (multi-row header),
        # false otherwise (including no header).
        if self.header is None:
            return False

        header = self.header
        if isinstance(header, (list, tuple, np.ndarray)):
            return len(header) > 1
        else:
            return False

    def _infer_columns(
        self,
    ) -> tuple[list[list[Scalar | None]], int, set[Scalar | None]]:
        # [FUNCTION DOC]
        # HELPER: reads the header line(s) and determines the raw columns.
        # Returns (list of column-label lists, one per header row; original
        # column count from the first header row; set of placeholder names given
        # to blank header cells). Consumes lines from the source; pushed-back
        # lookahead stays in the buffer (cleared unless a multi-row header ended
        # with a non-header extra line).
        # With a header: for each header row index, advances to that line. If the
        # file ends early: raises a value error stating the header spec and the
        # actual number of lines when some lines existed (excluding the
        # multi-row case's extra line); for a multi-row header with a later row
        # missing, appends a row of none labels and returns early; if no lines
        # at all, uses the user names or raises an empty-data error when there
        # are none. Blank cells become placeholder names that include the
        # position (and header level for multi-row headers). For single-row
        # headers, duplicates are renamed by appending a counter, handling
        # named columns before placeholders; if a dtype mapping covers the
        # original name it is copied to the renamed one (mutating the dtype
        # option). For multi-row headers the extra last row is probed to see
        # whether it holds index names; if not in that format it is replaced by
        # none labels and kept in the buffer.
        # After the header: with user names, reads the first data row to check
        # length, raises a value error if more names than both header fields and
        # first data row width, raises a type error if names are combined with a
        # multi-row header, applies column selection and uses the user names
        # (restricted to selected positions when the counts differ); without
        # user names applies column selection to the header columns.
        # Without a header: uses the first data line width as the column count;
        # user names or numeric positions become the columns, with checks that a
        # non-callable column selection length matches the names length (value
        # error otherwise).
        # KNOWN ISSUE: a source comment notes the duplicate-name logic should be
        # replaced by the shared deduplication helper.
        names = self.names
        num_original_columns = 0
        clear_buffer = True
        unnamed_cols: set[Scalar | None] = set()

        if self.header is not None:
            header = self.header
            have_mi_columns = self._have_mi_columns

            if isinstance(header, (list, tuple, np.ndarray)):
                # we have a mi columns, so read an extra line
                if have_mi_columns:
                    header = [*list(header), header[-1] + 1]
            else:
                header = [header]

            columns: list[list[Scalar | None]] = []
            for level, hr in enumerate(header):
                try:
                    line = self._buffered_line()

                    while self.line_pos <= hr:
                        line = self._next_line()

                except StopIteration as err:
                    if 0 < self.line_pos <= hr and (
                        not have_mi_columns or hr != header[-1]
                    ):
                        # If no rows we want to raise a different message and if
                        # we have mi columns, the last line is not part of the header
                        joi = list(map(str, header[:-1] if have_mi_columns else header))
                        msg = f"[{','.join(joi)}], len of {len(joi)}, "
                        raise ValueError(
                            f"Passed header={msg}but only {self.line_pos} lines in file"
                        ) from err

                    # We have an empty file, so check
                    # if columns are provided. That will
                    # serve as the 'line' for parsing
                    if have_mi_columns and hr > 0:
                        if clear_buffer:
                            self.buf.clear()
                        columns.append([None] * len(columns[-1]))
                        return columns, num_original_columns, unnamed_cols

                    if not self.names:
                        raise EmptyDataError("No columns to parse from file") from err

                    line = self.names[:]

                this_columns: list[Scalar | None] = []
                this_unnamed_cols = []

                for i, c in enumerate(line):
                    if c == "":
                        if have_mi_columns:
                            col_name = f"Unnamed: {i}_level_{level}"
                        else:
                            col_name = f"Unnamed: {i}"

                        this_unnamed_cols.append(i)
                        this_columns.append(col_name)
                    else:
                        this_columns.append(c)

                if not have_mi_columns:
                    counts: DefaultDict = defaultdict(int)
                    # Ensure that regular columns are used before unnamed ones
                    # to keep given names and mangle unnamed columns
                    col_loop_order = [
                        i
                        for i in range(len(this_columns))
                        if i not in this_unnamed_cols
                    ] + this_unnamed_cols

                    # TODO: Use pandas.io.common.dedup_names instead (see #50371)
                    for i in col_loop_order:
                        col = this_columns[i]
                        old_col = col
                        cur_count = counts[col]

                        if cur_count > 0:
                            while cur_count > 0:
                                counts[old_col] = cur_count + 1
                                col = f"{old_col}.{cur_count}"
                                if col in this_columns:
                                    cur_count += 1
                                else:
                                    cur_count = counts[col]

                            if (
                                self.dtype is not None
                                and is_dict_like(self.dtype)
                                and self.dtype.get(old_col) is not None
                                and self.dtype.get(col) is None
                            ):
                                self.dtype.update({col: self.dtype.get(old_col)})
                        this_columns[i] = col
                        counts[col] = cur_count + 1
                elif have_mi_columns:
                    # if we have grabbed an extra line, but it's not in our
                    # format so save in the buffer, and create a blank extra
                    # line for the rest of the parsing code
                    if hr == header[-1]:
                        lc = len(this_columns)
                        sic = self.index_col
                        ic = len(sic) if sic is not None else 0
                        unnamed_count = len(this_unnamed_cols)

                        # if wrong number of blanks or no index, not our format
                        if (lc != unnamed_count and lc - ic > unnamed_count) or ic == 0:
                            clear_buffer = False
                            this_columns = [None] * lc
                            self.buf = [self.buf[-1]]

                columns.append(this_columns)
                unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})

                if len(columns) == 1:
                    num_original_columns = len(this_columns)

            if clear_buffer:
                self.buf.clear()

            first_line: list[Scalar] | None
            if names is not None:
                # Read first row after header to check if data are longer
                try:
                    first_line = self._next_line()
                except StopIteration:
                    first_line = None

                len_first_data_row = 0 if first_line is None else len(first_line)

                if len(names) > len(columns[0]) and len(names) > len_first_data_row:
                    raise ValueError(
                        "Number of passed names did not match "
                        "number of header fields in the file"
                    )
                if len(columns) > 1:
                    raise TypeError("Cannot pass names with multi-index columns")

                if self.usecols is not None:
                    # Set _use_cols. We don't store columns because they are
                    # overwritten.
                    self._handle_usecols(columns, names, num_original_columns)
                else:
                    num_original_columns = len(names)
                if self._col_indices is not None and len(names) != len(
                    self._col_indices
                ):
                    columns = [[names[i] for i in sorted(self._col_indices)]]
                else:
                    columns = [names]
            else:
                columns = self._handle_usecols(
                    columns, columns[0], num_original_columns
                )
        else:
            ncols = len(self._header_line)
            num_original_columns = ncols

            if not names:
                columns = [list(range(ncols))]
                columns = self._handle_usecols(columns, columns[0], ncols)
            elif self.usecols is None or len(names) >= ncols:
                columns = self._handle_usecols([names], names, ncols)
                num_original_columns = len(names)
            elif not callable(self.usecols) and len(names) != len(self.usecols):
                raise ValueError(
                    "Number of passed names did not match number of "
                    "header fields in the file"
                )
            else:
                # Ignore output but set used columns.
                columns = [names]
                self._handle_usecols(columns, columns[0], ncols)

        return columns, num_original_columns, unnamed_cols

    @cache_readonly
    def _header_line(self):
        # [FUNCTION DOC]
        # HELPER: cached read-only property giving the first data line when the
        # header option is none (so that the width and index inference can reuse
        # it); returns none when a header exists. Takes the buffered line, or,
        # if the source is empty, a copy of the user names, and raises an
        # empty-data error when there are neither.
        # Store line for reuse in _get_index_name
        if self.header is not None:
            return None

        try:
            line = self._buffered_line()
        except StopIteration as err:
            if not self.names:
                raise EmptyDataError("No columns to parse from file") from err

            line = self.names[:]
        return line

    def _handle_usecols(
        self,
        columns: list[list[Scalar | None]],
        usecols_key: list[Scalar | None],
        num_original_columns: int,
    ) -> list[list[Scalar | None]]:
        # [FUNCTION DOC]
        # HELPER: applies the column selection to the header columns and records
        # the selected positions.
        # Inputs: columns list of label lists (one per header row); usecols_key
        # the labels used to look up string selections; num_original_columns the
        # field count for bounds checks. Without a selection returns the columns
        # unchanged. A callable selection yields positions from the key labels;
        # a selection containing strings requires a single header row (value
        # error otherwise) and maps names to positions, failing validation with
        # a value error for missing names, while integers in the mixed list are
        # used as positions; an all-integer selection raises a parser error if
        # any index is at or beyond the field count. Returns new label lists
        # containing only the selected positions, and stores the sorted
        # positions (side effect).
        """

        usecols_key is used if there are string usecols.
        """
        col_indices: set[int] | list[int]
        if self.usecols is not None:
            if callable(self.usecols):
                col_indices = evaluate_callable_usecols(self.usecols, usecols_key)
            elif any(isinstance(u, str) for u in self.usecols):
                if len(columns) > 1:
                    raise ValueError(
                        "If using multiple headers, usecols must be integers."
                    )
                col_indices = []

                for col in self.usecols:
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            self._validate_usecols_names(self.usecols, usecols_key)
                    else:
                        col_indices.append(col)
            else:
                missing_usecols = [
                    col for col in self.usecols if col >= num_original_columns
                ]
                if missing_usecols:
                    raise ParserError(
                        "Defining usecols with out-of-bounds indices is not allowed. "
                        f"{missing_usecols} are out-of-bounds.",
                    )
                col_indices = self.usecols

            columns = [
                [n for i, n in enumerate(column) if i in col_indices]
                for column in columns
            ]
            self._col_indices = sorted(col_indices)
        return columns

    def _buffered_line(self) -> list[Scalar]:
        # [FUNCTION DOC]
        # HELPER: returns the first buffered line without consuming it, reading
        # a new line (which also appends it to the buffer) when the buffer is
        # empty. May propagate end-of-iteration from the line reader.
        """
        Return a line from buffer, filling buffer if required.
        """
        if len(self.buf) > 0:
            return self.buf[0]
        else:
            return self._next_line()

    def _check_for_bom(self, first_row: list[Scalar]) -> list[Scalar]:
        # [FUNCTION DOC]
        # HELPER: removes a leading byte-order mark from the first row.
        # Returns the row unchanged when it is empty, its first element is not a
        # non-empty string, or that string does not start with the mark.
        # Otherwise returns a new row whose first element has the mark removed;
        # if the mark is immediately followed by the quote character, the
        # quoted content between the quotes (plus anything after the closing
        # quote) replaces the element.
        # KNOWN ISSUE: if a quote follows the mark but no closing quote exists in
        # the element, the search for the closing quote raises a value error.
        """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
        # first_row will be a list, so we need to check
        # that that list is not empty before proceeding.
        if not first_row:
            return first_row

        # The first element of this row is the one that could have the
        # BOM that we want to remove. Check that the first element is a
        # string before proceeding.
        if not isinstance(first_row[0], str):
            return first_row

        # Check that the string is not empty, as that would
        # obviously not have a BOM at the start of it.
        if not first_row[0]:
            return first_row

        # Since the string is non-empty, check that it does
        # in fact begin with a BOM.
        first_elt = first_row[0][0]
        if first_elt != _BOM:
            return first_row

        first_row_bom = first_row[0]
        new_row: str

        if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
            start = 2
            quote = first_row_bom[1]
            end = first_row_bom[2:].index(quote) + 2

            # Extract the data between the quotation marks
            new_row = first_row_bom[start:end]

            # Extract any remaining data after the second
            # quotation mark.
            if len(first_row_bom) > end + 1:
                new_row += first_row_bom[end + 1 :]

        else:
            # No quotation so just remove BOM from first element
            new_row = first_row_bom[1:]

        new_row_list: list[Scalar] = [new_row]
        return new_row_list + first_row[1:]

    def _is_line_empty(self, line: Sequence[Scalar]) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns true if the row has no fields or every field is falsy
        # (empty string, none, zero).
        """
        Check if a line is empty or not.

        Parameters
        ----------
        line : str, array-like
            The line of data to check.

        Returns
        -------
        boolean : Whether or not the line is empty.
        """
        return not line or all(not x for x in line)

    def _next_line(self) -> list[Scalar]:
        # [FUNCTION DOC]
        # HELPER: fetches the next logical row, appends it to the lookahead
        # buffer and returns it.
        # Skips rows selected by the skip predicate (advancing the position),
        # then reads rows, removes comment text, and drops blank rows when blank
        # skipping is on (otherwise keeps rows that were blank to begin with or
        # still have content after comment removal). For a nested-list source
        # exhaustion is signaled as end-of-iteration; for stream sources the
        # underlying iteration's own end-of-iteration is propagated, and rows
        # that fail to tokenize (reported or ignored per bad-line policy) are
        # skipped. When the row is the first of the file, a byte-order mark is
        # stripped. Increments the line counter.
        if isinstance(self.data, list):
            while self.skipfunc(self.pos):
                if self.pos >= len(self.data):
                    break
                self.pos += 1

            while True:
                try:
                    line = self._check_comments([self.data[self.pos]])[0]
                    self.pos += 1
                    # either uncommented or blank to begin with
                    if not self.skip_blank_lines and (
                        self._is_line_empty(self.data[self.pos - 1]) or line
                    ):
                        break
                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                except IndexError as err:
                    raise StopIteration from err
        else:
            while self.skipfunc(self.pos):
                self.pos += 1
                next(self.data)

            while True:
                orig_line = self._next_iter_line(row_num=self.pos + 1)
                self.pos += 1

                if orig_line is not None:
                    line = self._check_comments([orig_line])[0]

                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])

                        if ret:
                            line = ret[0]
                            break
                    elif self._is_line_empty(orig_line) or line:
                        break

        # This was the first line of the file,
        # which could contain the BOM at the
        # beginning of it.
        if self.pos == 1:
            line = self._check_for_bom(line)

        self.line_pos += 1
        self.buf.append(line)
        return line

    def _alert_malformed(self, msg: str, row_num: int) -> None:
        # [FUNCTION DOC]
        # HELPER: reports a malformed row according to the bad-line policy.
        # Inputs: msg the message; row_num the one-based row number for display.
        # Error policy: raises a parser error with the message. Warn policy or a
        # callable policy: emits a parser warning saying the line is skipped,
        # with the message. Skip policy: does nothing.
        # KNOWN ISSUE: the docstring mentions only the error and warn cases; a
        # callable policy also produces the warning.
        """
        Alert a user about a malformed row, depending on value of
        `self.on_bad_lines` enum.

        If `self.on_bad_lines` is ERROR, the alert will be `ParserError`.
        If `self.on_bad_lines` is WARN, the alert will be printed out.

        Parameters
        ----------
        msg: str
            The error message to display.
        row_num: int
            The row number where the parsing error occurred.
            Because this row number is displayed, we 1-index,
            even though we 0-index internally.
        """
        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
            raise ParserError(msg)
        if self.on_bad_lines == self.BadLineHandleMethod.WARN or callable(
            self.on_bad_lines
        ):
            warnings.warn(
                f"Skipping line {row_num}: {msg}\n",
                ParserWarning,
                stacklevel=find_stack_level(),
            )

    def _next_iter_line(self, row_num: int) -> list[Scalar] | None:
        # [FUNCTION DOC]
        # HELPER: takes the next row from the stream source.
        # Input row_num is the one-based number of the row, for messages.
        # Returns the row, or none if the tokenizer failed on it and the policy
        # allows continuing. A csv tokenization error under the error or warn
        # policy is reported via the malformed-row alert (which raises for
        # error), with a clearer message for null bytes and an extra hint when a
        # footer is being skipped; under skip or callable policies the row is
        # silently dropped. End-of-iteration propagates. Requires a stream
        # source (asserts it is not a list).
        """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num: int
            The row number of the line being parsed.
        """
        try:
            assert not isinstance(self.data, list)
            line = next(self.data)
            # lie about list[str] vs list[Scalar] to minimize ignores
            return line  # type: ignore[return-value]
        except csv.Error as e:
            if self.on_bad_lines in (
                self.BadLineHandleMethod.ERROR,
                self.BadLineHandleMethod.WARN,
            ):
                msg = str(e)

                if "NULL byte" in msg or "line contains NUL" in msg:
                    msg = (
                        "NULL byte detected. This byte "
                        "cannot be processed in Python's "
                        "native csv library at the moment, "
                        "so please pass in engine='c' instead"
                    )

                if self.skipfooter > 0:
                    reason = (
                        "Error could possibly be due to "
                        "parsing errors in the skipped footer rows "
                        "(the skipfooter keyword is only applied "
                        "after Python's csv library has parsed "
                        "all rows)."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # HELPER: strips comment text from rows. Without a comment character
        # returns the input unchanged. Otherwise returns new rows: each field is
        # kept whole if it is not a string, lacks the comment character, or is
        # one of the missing-value markers; the first field containing the
        # comment character is truncated at it (kept only if non-empty) and the
        # rest of the row is dropped.
        if self.comment is None:
            return lines
        ret = []
        for line in lines:
            rl = []
            for x in line:
                if (
                    not isinstance(x, str)
                    or self.comment not in x
                    or x in self.na_values
                ):
                    rl.append(x)
                else:
                    x = x[: x.find(self.comment)]
                    if len(x) > 0:
                        rl.append(x)
                    break
            ret.append(rl)
        return ret

    def _remove_empty_lines(self, lines: list[list[T]]) -> list[list[T]]:
        # [FUNCTION DOC]
        # HELPER: returns a new list of rows without blank rows, where blank
        # means no fields, or a single field that is a string consisting only of
        # whitespace. Rows with several fields, or a single non-string field,
        # are kept. Overridden for fixed-width input.
        """
        Iterate through the lines and remove any that are
        either empty or contain only one whitespace value

        Parameters
        ----------
        lines : list of list of Scalars
            The array of lines that we are to filter.

        Returns
        -------
        filtered_lines : list of list of Scalars
            The same array of lines with the "empty" ones removed.
        """
        # Remove empty lines and lines with only one whitespace value
        ret = [
            line
            for line in lines
            if (
                len(line) > 1
                or (
                    len(line) == 1 and (not isinstance(line[0], str) or line[0].strip())
                )
            )
        ]
        return ret

    def _check_thousands(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # HELPER: removes the thousands separator from numeric-looking text
        # cells. Returns the rows unchanged when no separator is configured,
        # otherwise the result of the shared replacement routine with the
        # separator replaced by nothing.
        if self.thousands is None:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.thousands, replace=""
        )

    def _search_replace_num_columns(
        self, lines: list[list[Scalar]], search: str, replace: str
    ) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # HELPER: shared routine for thousands and decimal handling. Inputs:
        # lines rows of fields; search the marker to replace; replace its
        # replacement. Returns new rows in which a field is rewritten (all
        # occurrences of the marker replaced) only if it is a string containing
        # the marker, its column is not excluded from this treatment, and its
        # stripped text matches the numeric pattern; other fields are kept.
        ret = []
        for line in lines:
            rl = []
            for i, x in enumerate(line):
                if (
                    not isinstance(x, str)
                    or search not in x
                    or i in self._no_thousands_columns
                    or not self.num.search(x.strip())
                ):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # HELPER: normalizes a custom decimal marker to a period in
        # numeric-looking text cells. Returns the rows unchanged when the marker
        # already is the default period; otherwise the result of the shared
        # replacement routine.
        if self.decimal == parser_defaults["decimal"]:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.decimal, replace="."
        )

    def _get_index_name(
        self,
    ) -> tuple[Sequence[Hashable] | None, list[Hashable], list[Hashable]]:
        # [FUNCTION DOC]
        # HELPER: determines index names and finalizes original names/columns.
        # Returns (index names or none, original names, columns). Reads the
        # first data line (the stored header line or a freshly read one) and
        # one more lookahead line. Case 1 (implicit index): more fields in the
        # first data line than original columns (unless the index option is
        # false) means leading columns form the index; sets the implicit flag,
        # defaults the index columns to the leading positions and returns no
        # names. Case 0 (names on separate rows): with a header, the index not
        # false, and the next line's width equal to the first line's width plus
        # the original column count, the first line holds index names: sets the
        # index columns to those positions, drops one buffered line, prepends
        # the index names to the columns and original names, updates the
        # original column count, and returns early with that line as the index
        # names. Case 2: otherwise cleans index names from the columns per the
        # index option (updating it). Side effects on buffer, flags and index
        # option as described.
        """
        Try several cases to get lines:

        0) There are headers on row 0 and row 1 and their
        total summed lengths equals the length of the next line.
        Treat row 0 as columns and row 1 as indices
        1) Look for implicit index: there are more columns
        on row 1 than row 0. If this is true, assume that row
        1 lists index columns and row 0 lists normal columns.
        2) Get index from the columns if it was listed.
        """
        columns: Sequence[Hashable] = self.orig_names
        orig_names = list(columns)
        columns = list(columns)

        line: list[Scalar] | None
        if self._header_line is not None:
            line = self._header_line
        else:
            try:
                line = self._next_line()
            except StopIteration:
                line = None

        next_line: list[Scalar] | None
        try:
            next_line = self._next_line()
        except StopIteration:
            next_line = None

        # implicitly index_col=0 b/c 1 fewer column names
        implicit_first_cols = 0
        if line is not None:
            # leave it 0, #2442
            # Case 1
            index_col = self.index_col
            if index_col is not False:
                implicit_first_cols = len(line) - self.num_original_columns

            # Case 0
            if (
                next_line is not None
                and self.header is not None
                and index_col is not False
            ):
                if len(next_line) == len(line) + self.num_original_columns:
                    # column and index names on diff rows
                    self.index_col = list(range(len(line)))
                    self.buf = self.buf[1:]

                    for c in reversed(line):
                        columns.insert(0, c)

                    # Update list of original names to include all indices.
                    orig_names = list(columns)
                    self.num_original_columns = len(columns)
                    return line, orig_names, columns

        if implicit_first_cols > 0:
            # Case 1
            self._implicit_index = True
            if self.index_col is None:
                self.index_col = list(range(implicit_first_cols))

            index_name = None

        else:
            # Case 2
            (index_name, _, self.index_col) = self._clean_index_names(
                columns, self.index_col
            )

        return index_name, orig_names, columns

    def _rows_to_cols(self, content: list[list[Scalar]]) -> list[np.ndarray]:
        # [FUNCTION DOC]
        # HELPER: transposes row lists into column arrays, enforcing width.
        # Input content is a non-empty list of rows (an empty list fails when
        # taking the maximum width). Expected width is the original column count
        # plus the index columns when an implicit index is used. If some row is
        # wider than expected, and the index option is not false and no column
        # selection is set, each wide row is handled per bad-line policy: with
        # a callable, the callable receives the row and may return a replacement
        # (truncated if still too wide, recorded as a callable-source problem)
        # or none to drop it; with error or warn policies the row is recorded
        # (the error policy stops at the first) and dropped; narrow rows are kept.
        # Then each recorded problem is reported via the malformed-row alert with
        # a message giving the expected width, row number and actual width, plus
        # a hint about quoting for multi-character delimiters. The rows are
        # converted to an object matrix padded to the expected width and
        # transposed to a list of column arrays; with a column selection, only
        # the index columns (if implicit) and selected positions are kept.
        # Under the skip policy wide rows are dropped silently. Row number
        # accounting subtracts the pending rows and footer length from the
        # current position.
        col_len = self.num_original_columns

        if self._implicit_index:
            col_len += len(self.index_col)

        max_len = max(len(row) for row in content)

        # Check that there are no rows with too many
        # elements in their row (rows with too few
        # elements are padded with NaN).
        if max_len > col_len and self.index_col is not False and self.usecols is None:
            footers = self.skipfooter if self.skipfooter else 0
            bad_lines = []

            iter_content = enumerate(content)
            content_len = len(content)
            content = []

            for i, _content in iter_content:
                actual_len = len(_content)
                if actual_len > col_len:
                    if callable(self.on_bad_lines):
                        new_l = self.on_bad_lines(_content)
                        if new_l is not None:
                            new_l = cast(list[Scalar], new_l)
                            if len(new_l) > col_len:
                                row_num = self.pos - (content_len - i + footers)
                                bad_lines.append((row_num, len(new_l), "callable"))
                                new_l = new_l[:col_len]
                            content.append(new_l)

                    elif self.on_bad_lines in (
                        self.BadLineHandleMethod.ERROR,
                        self.BadLineHandleMethod.WARN,
                    ):
                        row_num = self.pos - (content_len - i + footers)
                        bad_lines.append((row_num, actual_len, "normal"))
                        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
                            break
                else:
                    content.append(_content)

            for row_num, actual_len, source in bad_lines:
                msg = (
                    f"Expected {col_len} fields in line {row_num + 1}, saw {actual_len}"
                )
                if source == "callable":
                    msg += " from bad_lines callable"
                elif (
                    self.delimiter
                    and len(self.delimiter) > 1
                    and self.quoting != csv.QUOTE_NONE
                ):
                    # see gh-13374
                    reason = (
                        "Error could possibly be due to quotes being "
                        "ignored when a multi-char delimiter is used."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num + 1)

        # see gh-13320
        zipped_content = list(lib.to_object_array(content, min_width=col_len).T)

        if self.usecols:
            assert self._col_indices is not None
            col_indices = self._col_indices

            if self._implicit_index:
                zipped_content = [
                    a
                    for i, a in enumerate(zipped_content)
                    if (
                        i < len(self.index_col)
                        or i - len(self.index_col) in col_indices
                    )
                ]
            else:
                zipped_content = [
                    a for i, a in enumerate(zipped_content) if i in col_indices
                ]
        return zipped_content

    def _get_lines(self, rows: int | None = None) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # HELPER: gathers up to rows data rows (all when none), cleaned.
        # Takes rows from the lookahead buffer first, then from the source (list
        # slice for nested lists; stream iteration otherwise, counting only
        # non-skipped rows toward the requested number), applies row skipping,
        # advances the position counter, and clears the buffer when source rows
        # were read. Raises end-of-iteration when the source is exhausted and
        # nothing was gathered. Then drops footer rows, strips comments, drops
        # blank rows if configured, removes thousands separators and normalizes
        # the decimal marker. Returns the resulting rows.
        # KNOWN ISSUE: the footer rows are cut from every chunk returned, not
        # only the last one, so chunked reading with a footer count drops that
        # many rows from each chunk.
        # KNOWN ISSUE: the offset computation checks the position against none
        # although the position is always an integer (dead branch).
        lines = self.buf
        new_rows = None

        # already fetched some number
        if rows is not None:
            # we already have the lines in the buffer
            if len(self.buf) >= rows:
                new_rows, self.buf = self.buf[:rows], self.buf[rows:]

            # need some lines
            else:
                rows -= len(self.buf)

        if new_rows is None:
            if isinstance(self.data, list):
                if self.pos > len(self.data):
                    raise StopIteration
                if rows is None:
                    new_rows = self.data[self.pos :]
                    new_pos = len(self.data)
                else:
                    new_rows = self.data[self.pos : self.pos + rows]
                    new_pos = self.pos + rows

                new_rows = self._remove_skipped_rows(new_rows)
                lines.extend(new_rows)
                self.pos = new_pos

            else:
                new_rows = []
                try:
                    if rows is not None:
                        row_index = 0
                        row_ct = 0
                        offset = self.pos if self.pos is not None else 0
                        while row_ct < rows:
                            new_row = next(self.data)
                            if not self.skipfunc(offset + row_index):
                                row_ct += 1
                            row_index += 1
                            new_rows.append(new_row)

                        len_new_rows = len(new_rows)
                        new_rows = self._remove_skipped_rows(new_rows)
                        lines.extend(new_rows)
                    else:
                        rows = 0

                        while True:
                            next_row = self._next_iter_line(row_num=self.pos + rows + 1)
                            rows += 1

                            if next_row is not None:
                                new_rows.append(next_row)
                        len_new_rows = len(new_rows)

                except StopIteration:
                    len_new_rows = len(new_rows)
                    new_rows = self._remove_skipped_rows(new_rows)
                    lines.extend(new_rows)
                    if len(lines) == 0:
                        raise
                self.pos += len_new_rows

            self.buf = []
        else:
            lines = new_rows

        if self.skipfooter:
            lines = lines[: -self.skipfooter]

        lines = self._check_comments(lines)
        if self.skip_blank_lines:
            lines = self._remove_empty_lines(lines)
        lines = self._check_thousands(lines)
        return self._check_decimal(lines)

    def _remove_skipped_rows(self, new_rows: list[list[Scalar]]) -> list[list[Scalar]]:
        # [FUNCTION DOC]
        # HELPER: when a skip-rows option is set (truthy), returns the rows whose
        # absolute position (index within the batch plus the current position)
        # does not satisfy the skip predicate; otherwise returns the rows
        # unchanged.
        if self.skiprows:
            return [
                row for i, row in enumerate(new_rows) if not self.skipfunc(i + self.pos)
            ]
        return new_rows

    def _set_no_thousand_columns(self) -> set[int]:
        # [FUNCTION DOC]
        # HELPER: computes positions of columns where thousands/decimal rewriting
        # must not be applied. Includes date-parsed columns (when columns exist
        # and date parsing is on, via the base-class helper) and, when a dtype
        # option exists, columns whose target dtype is not numeric (or is
        # boolean): for a single dtype all columns if it is non-numeric, for a
        # mapping the listed columns that qualify. Returns the set of positions.
        no_thousands_columns: set[int] = set()
        if self.columns and self.parse_dates:
            assert self._col_indices is not None
            no_thousands_columns = self._set_noconvert_dtype_columns(
                self._col_indices, self.columns
            )
        if self.columns and self.dtype:
            assert self._col_indices is not None
            for i, col in zip(self._col_indices, self.columns, strict=True):
                if not isinstance(self.dtype, dict) and not is_numeric_dtype(
                    self.dtype
                ):
                    no_thousands_columns.add(i)
                if (
                    isinstance(self.dtype, dict)
                    and col in self.dtype
                    and (
                        not is_numeric_dtype(self.dtype[col])
                        or is_bool_dtype(self.dtype[col])
                    )
                ):
                    no_thousands_columns.add(i)
        return no_thousands_columns


class FixedWidthReader(abc.Iterator):
    # [CLASS DOC]
    # Iterator over a fixed-width text stream that yields, for each line, the
    # list of stripped field strings cut out by column ranges. State: f the
    # stream, buffer an optional iterator of lines consumed while inferring
    # column ranges (replayed before continuing the stream), delimiter the set
    # of characters stripped from field edges (whitespace by default, or line
    # breaks plus the user delimiter), comment, colspecs the list of half-open
    # (start, end) ranges. Implements the standard iterator protocol.
    """
    A reader of fixed-width lines.
    """

    def __init__(
        self,
        f: IO[str] | ReadCsvBuffer[str],
        colspecs: list[tuple[int, int]] | Literal["infer"],
        delimiter: str | None,
        comment: str | None,
        skiprows: set[int] | None = None,
        infer_nrows: int = 100,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs: f the stream; colspecs a list of (start, end) pairs or the
        # word infer; delimiter an extra character set to strip (none means strip
        # whitespace-like characters); comment the comment marker or none;
        # skiprows a set of row numbers to ignore while inferring (optional);
        # infer_nrows the number of sample rows for inference (default 100).
        # Builds the strip set, infers the ranges when requested (consuming and
        # buffering sample lines), and validates the ranges: raises a type error
        # if they are not a list or tuple, or if any entry is not a two-element
        # list or tuple whose items are integers or none.
        self.f = f
        self.buffer: Iterator | None = None
        self.delimiter = "\r\n" + delimiter if delimiter else "\n\r\t "
        self.comment = comment
        if colspecs == "infer":
            self.colspecs = self.detect_colspecs(
                infer_nrows=infer_nrows, skiprows=skiprows
            )
        else:
            self.colspecs = colspecs

        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(
                "column specifications must be a list or tuple, "
                f"input was a {type(colspecs).__name__}"
            )

        for colspec in self.colspecs:
            if not (
                isinstance(colspec, (tuple, list))
                and len(colspec) == 2
                and isinstance(colspec[0], (int, np.integer, type(None)))
                and isinstance(colspec[1], (int, np.integer, type(None)))
            ):
                raise TypeError(
                    "Each column specification must be "
                    "2 element tuple or list of integers"
                )

    def get_rows(self, infer_nrows: int, skiprows: set[int] | None = None) -> list[str]:
        # [FUNCTION DOC]
        # Reads lines from the stream for column inference. Reads until
        # infer_nrows lines that are not in the skip set have been collected (or
        # the stream ends). Every line read, skipped or not, is also stored so
        # the main iteration can replay it (the buffer attribute is set,
        # consuming the stream by side effect). Returns only the non-skipped
        # lines.
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        if skiprows is None:
            skiprows = set()
        buffer_rows = []
        detect_rows = []
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)
            buffer_rows.append(row)
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(
        self, infer_nrows: int = 100, skiprows: set[int] | None = None
    ) -> list[tuple[int, int]]:
        # [FUNCTION DOC]
        # Infers field ranges from sample lines. Inputs: infer_nrows number of
        # lines to sample (default 100); skiprows lines to ignore. Reads the
        # sample (raising an empty-data error if there are no lines), strips
        # comments from each sample line if a comment marker is set, marks every
        # character position covered by a run of non-delimiter characters in
        # any sample line, and returns the start/end pairs of the marked runs
        # (half-open). Positions with no content in any sample separate fields.
        # Regex escape the delimiters
        delimiters = "".join([rf"\{x}" for x in self.delimiter])
        pattern = re.compile(f"([^{delimiters}]+)")
        rows = self.get_rows(infer_nrows, skiprows)
        if not rows:
            raise EmptyDataError("No rows from which to infer column width")
        max_len = max(map(len, rows))
        mask = np.zeros(max_len + 1, dtype=int)
        if self.comment is not None:
            rows = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start() : m.end()] = 1
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        edges = np.where((mask ^ shifted) == 1)[0]
        edge_pairs = list(zip(edges[::2], edges[1::2], strict=True))
        return edge_pairs

    def __next__(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the next line cut into fields: takes lines from the replay
        # buffer first (discarding it when exhausted), then from the stream;
        # end-of-iteration propagates. Each field is the slice over its
        # half-open range, with the strip set removed from both ends.
        if self.buffer is not None:
            try:
                line = next(self.buffer)
            except StopIteration:
                self.buffer = None
                line = next(self.f)  # type: ignore[arg-type]
        else:
            line = next(self.f)  # type: ignore[arg-type]
        # Note: 'colspecs' is a sequence of half-open intervals.
        return [line[from_:to].strip(self.delimiter) for (from_, to) in self.colspecs]


class FixedWidthFieldParser(PythonParser):
    # [CLASS DOC]
    # Fixed-width variant of the Python parsing engine. Extends PythonParser by
    # swapping the row source for a fixed-width reader and by redefining blank
    # row detection. Extra state: colspecs (column ranges or the word infer) and
    # infer_nrows (sample size for inference).
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None:
        # [FUNCTION DOC]
        # Removes the column-range and sample-size options from the option
        # mapping (mutating it; both keys are required), stores them, then runs
        # the parent initialization, which calls back into the reader factory
        # below.
        # Support iterators, convert to a list.
        self.colspecs = kwds.pop("colspecs")
        self.infer_nrows = kwds.pop("infer_nrows")
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f: IO[str] | ReadCsvBuffer[str]) -> FixedWidthReader:
        # [FUNCTION DOC]
        # HELPER: override of the reader factory; builds a fixed-width reader on
        # the stream using the stored column ranges, delimiter, comment,
        # skip-rows spec and sample size.
        return FixedWidthReader(
            f,
            self.colspecs,
            self.delimiter,
            self.comment,
            self.skiprows,
            self.infer_nrows,
        )

    def _remove_empty_lines(self, lines: list[list[T]]) -> list[list[T]]:
        # [FUNCTION DOC]
        # HELPER: override of blank row removal for fixed-width rows. Returns the
        # rows that have at least one field that is either not a string or a
        # string with non-whitespace content (blank rows are lists of empty
        # strings, so a length test would not work).
        """
        Returns the list of lines without the empty ones. With fixed-width
        fields, empty lines become arrays of empty strings.

        See PythonParser._remove_empty_lines.
        """
        return [
            line
            for line in lines
            if any(not isinstance(e, str) or e.strip() for e in line)
        ]


def _validate_skipfooter_arg(skipfooter: int) -> int:
    # [FUNCTION DOC]
    # HELPER: validates the footer-row count for the parser constructor.
    # Raises a value error if it is not an integer, or if it is negative;
    # otherwise returns it unchanged.
    """
    Validate the 'skipfooter' parameter.

    Checks whether 'skipfooter' is a non-negative integer.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    skipfooter : non-negative integer
        The number of rows to skip at the end of the file.

    Returns
    -------
    validated_skipfooter : non-negative integer
        The original input if the validation succeeds.

    Raises
    ------
    ValueError : 'skipfooter' was not a non-negative integer.
    """
    if not is_integer(skipfooter):
        raise ValueError("skipfooter must be an integer")

    if skipfooter < 0:
        raise ValueError("skipfooter cannot be negative")

    # Incompatible return value type (got "Union[int, integer[Any]]", expected "int")
    return skipfooter  # type: ignore[return-value]
