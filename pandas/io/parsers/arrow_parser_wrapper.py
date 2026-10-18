# [MODULE DOC]
# Adapter that lets the delimited-text reader use the pyarrow CSV reader as an
# engine. Public content: the ArrowParserWrapper class. Relationships: a
# subclass of the shared parser base; created by the readers module when the
# pyarrow engine is selected; translates pandas reader options into pyarrow
# read, parse and convert options and post-processes the arrow table into a
# frame (dates, index, dtype). pyarrow is imported lazily. No import-time
# side effects beyond imports.
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    Pandas4Warning,
    ParserError,
    ParserWarning,
)
from pandas.util._exceptions import (
    find_stack_level,
)

from pandas.core.dtypes.common import (
    pandas_dtype,
)
from pandas.core.dtypes.inference import is_integer

from pandas.io._util import arrow_table_to_pandas
from pandas.io.parsers.base_parser import ParserBase

if TYPE_CHECKING:
    import pyarrow as pa

    from pandas._typing import ReadBuffer

    from pandas import DataFrame


class ArrowParserWrapper(ParserBase):
    # [CLASS DOC]
    # Parsing engine for the delimited reader backed by pyarrow. Extends the
    # shared parser base. State: kwds (option mapping, later modified with
    # pyarrow-style names), src (binary source), encoding (defaults to utf-8),
    # na_values (list), and after option translation parse_options,
    # convert_options and read_options mappings. Single-shot: read parses the
    # whole source into a frame. Many pandas options are unsupported and are
    # rejected earlier by the readers module.
    """
    Wrapper for the pyarrow engine for read_csv()
    """

    def __init__(self, src: ReadBuffer[bytes], **kwds) -> None:
        # [FUNCTION DOC]
        # Stores the binary source src and the option mapping kwds, runs the
        # base-class initialization on the mapping (which removes some date
        # options from it), then validates keywords. Returns nothing; may raise
        # whatever base initialization or validation raises.
        super().__init__(kwds)
        self.kwds = kwds
        self.src = src

        self._parse_kwds()

    def _parse_kwds(self) -> None:
        # [FUNCTION DOC]
        # HELPER: validates and normalizes keywords before translation. Sets
        # encoding to the given value or utf-8 when absent, and na_values to a
        # list copy of the given missing-value collection. Raises a value error
        # when na_values is a mapping (per-column values unsupported by this
        # engine).
        """
        Validates keywords before passing to pyarrow.
        """
        encoding: str | None = self.kwds.get("encoding")
        self.encoding = "utf-8" if encoding is None else encoding

        na_values = self.kwds["na_values"]
        if isinstance(na_values, dict):
            raise ValueError(
                "The pyarrow engine doesn't support passing a dict for na_values"
            )
        self.na_values = list(self.kwds["na_values"])

    def _get_pyarrow_options(self) -> None:
        # [FUNCTION DOC]
        # HELPER: translates stored pandas options into pyarrow option mappings.
        # Mutates the keyword mapping: renames (only when the value is not none)
        # column selection to include-columns, missing values to null values,
        # escape character, skip-blank-lines to ignore-empty-lines, decimal to
        # decimal point and quote character to quote char. The date format is
        # turned into a one-element list of timestamp parsers when it is a
        # string; any other value (including a per-column mapping) becomes none,
        # so mapping formats are not passed on.
        # Builds parse_options from non-none values of delimiter, quote char,
        # escape char and ignore-empty-lines, plus an invalid-row handler per
        # the bad-lines policy: a callable is passed through; the error policy
        # sets no handler (arrow raises); warn installs a nested handler that
        # emits a parser warning with expected and actual column counts and the
        # row text and returns a skip action; skip installs a handler returning
        # skip. Builds convert_options from non-none values of include-columns,
        # null values, true/false values, decimal point, timestamp parsers, and
        # sets strings-can-be-null when the empty string is among the null
        # values. When there is no header, include-columns are converted to
        # the auto-generated names (letter f plus number). Builds read_options:
        # auto-generate names when the header is none; rows to skip equal to the
        # header value when a header is given (so the header row becomes the
        # first row read) else the skiprows keyword; and the encoding.
        """
        Rename some arguments to pass to pyarrow
        """
        mapping = {
            "usecols": "include_columns",
            "na_values": "null_values",
            "escapechar": "escape_char",
            "skip_blank_lines": "ignore_empty_lines",
            "decimal": "decimal_point",
            "quotechar": "quote_char",
        }
        for pandas_name, pyarrow_name in mapping.items():
            if pandas_name in self.kwds and self.kwds.get(pandas_name) is not None:
                self.kwds[pyarrow_name] = self.kwds.pop(pandas_name)

        # Date format handling
        # If we get a string, we need to convert it into a list for pyarrow
        # If we get a dict, we want to parse those separately
        date_format = self.date_format
        if isinstance(date_format, str):
            date_format = [date_format]
        else:
            # In case of dict, we don't want to propagate through, so
            # just set to pyarrow default of None

            # Ideally, in future we disable pyarrow dtype inference (read in as string)
            # to prevent misreads.
            date_format = None
        self.kwds["timestamp_parsers"] = date_format

        self.parse_options = {
            option_name: option_value
            for option_name, option_value in self.kwds.items()
            if option_value is not None
            and option_name
            in ("delimiter", "quote_char", "escape_char", "ignore_empty_lines")
        }

        on_bad_lines = self.kwds.get("on_bad_lines")
        if on_bad_lines is not None:
            if callable(on_bad_lines):
                self.parse_options["invalid_row_handler"] = on_bad_lines
            elif on_bad_lines == ParserBase.BadLineHandleMethod.ERROR:
                self.parse_options["invalid_row_handler"] = (
                    None  # PyArrow raises an exception by default
                )
            elif on_bad_lines == ParserBase.BadLineHandleMethod.WARN:

                def handle_warning(invalid_row) -> str:
                    # [FUNCTION DOC]
                    # HELPER: nested invalid-row callback given to pyarrow.
                    # Receives an object with expected column count, actual
                    # column count and row text; emits a parser warning
                    # describing them and returns the string telling pyarrow to
                    # skip the row.
                    warnings.warn(
                        f"Expected {invalid_row.expected_columns} columns, but found "
                        f"{invalid_row.actual_columns}: {invalid_row.text}",
                        ParserWarning,
                        stacklevel=find_stack_level(),
                    )
                    return "skip"

                self.parse_options["invalid_row_handler"] = handle_warning
            elif on_bad_lines == ParserBase.BadLineHandleMethod.SKIP:
                self.parse_options["invalid_row_handler"] = lambda _: "skip"

        self.convert_options = {
            option_name: option_value
            for option_name, option_value in self.kwds.items()
            if option_value is not None
            and option_name
            in (
                "include_columns",
                "null_values",
                "true_values",
                "false_values",
                "decimal_point",
                "timestamp_parsers",
            )
        }
        self.convert_options["strings_can_be_null"] = "" in self.kwds["null_values"]
        # autogenerated column names are prefixed with 'f' in pyarrow.csv
        if self.header is None and "include_columns" in self.convert_options:
            self.convert_options["include_columns"] = [
                f"f{n}" for n in self.convert_options["include_columns"]
            ]

        self.read_options = {
            "autogenerate_column_names": self.header is None,
            "skip_rows": self.header
            if self.header is not None
            else self.kwds["skiprows"],
            "encoding": self.encoding,
        }

    def _get_convert_options(self):
        # [FUNCTION DOC]
        # HELPER: builds the pyarrow convert-options object from the stored
        # convert option mapping (requires pyarrow). On a type error from
        # construction: re-validates the column selection (raising a value error
        # for non-string selections or callables), then, if the null values are
        # not a list-like of strings only, raises a type error saying the engine
        # requires all missing-value markers to be strings (chained to the
        # original); otherwise re-raises the original error. Returns the options
        # object.
        pyarrow_csv = import_optional_dependency("pyarrow.csv")

        try:
            convert_options = pyarrow_csv.ConvertOptions(**self.convert_options)
        except TypeError as err:
            include = self.convert_options.get("include_columns", None)
            if include is not None:
                self._validate_usecols(include)

            nulls = self.convert_options.get("null_values", set())
            if not lib.is_list_like(nulls) or not all(
                isinstance(x, str) for x in nulls
            ):
                raise TypeError(
                    "The 'pyarrow' engine requires all na_values to be strings"
                ) from err

            raise

        return convert_options

    def _adjust_column_names(self, table: pa.Table) -> bool:
        # [FUNCTION DOC]
        # HELPER: reconciles user-facing names with the arrow table width.
        # Input: table the parsed arrow table. Only acts when there is no
        # header: if names are unset, sets them to a range of the column count;
        # if the names length differs from the column count (only possible with
        # integer index columns), prepends placeholder string-number names so the
        # lengths match (mutating the names, which then must be a list). Returns
        # false when such padding happened (the index will have no name), else
        # true.
        num_cols = len(table.columns)
        multi_index_named = True
        if self.header is None:
            if self.names is None:
                self.names = range(num_cols)
            if len(self.names) != num_cols:
                # usecols is passed through to pyarrow, we only handle index col here
                # The only way self.names is not the same length as number of cols is
                # if we have int index_col. We should just pad the names(they will get
                # removed anyways) to expected length then.
                columns_prefix = [str(x) for x in range(num_cols - len(self.names))]
                self.names = columns_prefix + self.names
                multi_index_named = False
        return multi_index_named

    def _finalize_index(self, frame: DataFrame, multi_index_named: bool) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: applies the index column option to a parsed frame.
        # Inputs: frame the frame; multi_index_named false when index names are
        # placeholders. If no index column is set, returns the frame as is.
        # Otherwise, for each entry (integers resolved to column labels; a
        # string not in the columns raises a value error), if a dtype option
        # exists applies the matching dtype to that column first and removes
        # it from the dtype mapping so it is not applied again (mutating the
        # stored dtype), then sets the index in place from those columns,
        # dropping them. For headerless input with placeholder names, clears
        # the index names. Returns the frame.
        # KNOWN ISSUE: when a dtype option is set and a string index entry has
        # no dtype of its own, the fallback looks up the column by indexing the
        # column labels with that string, which fails; the dtype option is also
        # assumed to be a mapping here, so a single dtype value raises an
        # attribute error when an index column is set.
        if self.index_col is not None:
            index_to_set = self.index_col.copy()
            for i, item in enumerate(self.index_col):
                if is_integer(item):
                    index_to_set[i] = frame.columns[item]
                # String case
                elif item not in frame.columns:
                    raise ValueError(f"Index {item} invalid")

                # Process dtype for index_col and drop from dtypes
                if self.dtype is not None:
                    key, new_dtype = (
                        (item, self.dtype.get(item))
                        if self.dtype.get(item) is not None
                        else (frame.columns[item], self.dtype.get(frame.columns[item]))
                    )
                    if new_dtype is not None:
                        frame[key] = frame[key].astype(new_dtype)
                        del self.dtype[key]

            frame.set_index(index_to_set, drop=True, inplace=True)
            # Clear names if headerless and no name given
            if self.header is None and not multi_index_named:
                frame.index.names = [None] * len(frame.index.names)

        return frame

    def _finalize_dtype(self, frame: DataFrame) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: casts the frame to the dtype option. Does nothing if unset.
        # A mapping is reduced to entries for columns present (missing columns
        # are ignored, like other engines) with values resolved to dtype
        # objects; a single value is resolved once. Both replace the stored
        # dtype. Casts the frame; a type error from the cast is re-raised as a
        # value error with the same message. Returns the cast frame.
        if self.dtype is not None:
            # Ignore non-existent columns from dtype mapping
            # like other parsers do
            if isinstance(self.dtype, dict):
                self.dtype = {
                    k: pandas_dtype(v)
                    for k, v in self.dtype.items()
                    if k in frame.columns
                }
            else:
                self.dtype = pandas_dtype(self.dtype)
            try:
                frame = frame.astype(self.dtype)
            except TypeError as err:
                # GH#44901 reraise to keep api consistent
                raise ValueError(str(err)) from err
        return frame

    def _finalize_pandas_output(
        self, frame: DataFrame, multi_index_named: bool
    ) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: post-processing pipeline for the parsed frame: applies date
        # conversions to all columns, then the index option, then the dtype
        # option, in that order, and returns the result. multi_index_named is
        # forwarded to the index step.
        """
        Processes data read in based on kwargs.

        Parameters
        ----------
        frame : DataFrame
            The DataFrame to process.
        multi_index_named : bool

        Returns
        -------
        DataFrame
            The processed DataFrame.
        """
        frame = self._do_date_conversions(frame.columns, frame)
        frame = self._finalize_index(frame, multi_index_named)
        frame = self._finalize_dtype(frame)
        return frame

    def _validate_usecols(self, usecols) -> None:
        # [FUNCTION DOC]
        # HELPER: rejects column selections unsupported by this engine. Raises a
        # value error if the selection is a list-like containing any non-string
        # (integer positions), or if it is callable. Otherwise returns nothing.
        # Note this overrides nothing in the base class (the base has a
        # names-based validator with a different name).
        if lib.is_list_like(usecols) and not all(isinstance(x, str) for x in usecols):
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be integer "
                "column positions. Pass a list of string column names instead."
            )
        elif callable(usecols):
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be a callable."
            )

    def read(self) -> DataFrame:
        # [FUNCTION DOC]
        # Parses the whole source into a frame. Requires pyarrow. Steps:
        # translate options; build convert options; read the CSV with arrow (an
        # arrow invalid-data error becomes a parser error chained to it); when no
        # dtype backend is requested, casts null-typed arrow columns to 64-bit
        # float; adjusts names for headerless input; converts to a frame with a
        # block-construction deprecation warning suppressed, passing the
        # backend, a flag turning null columns into nullable 64-bit integers,
        # the dtype option and names; for headerless input sets the column
        # labels to the names; finally runs the post-processing pipeline.
        # Returns the frame.
        """
        Reads the contents of a CSV file into a DataFrame and
        processes it according to the kwargs passed in the
        constructor.

        Returns
        -------
        DataFrame
            The DataFrame created from the CSV file.
        """
        pa = import_optional_dependency("pyarrow")
        pyarrow_csv = import_optional_dependency("pyarrow.csv")
        self._get_pyarrow_options()
        convert_options = self._get_convert_options()

        try:
            table = pyarrow_csv.read_csv(
                self.src,
                read_options=pyarrow_csv.ReadOptions(**self.read_options),
                parse_options=pyarrow_csv.ParseOptions(**self.parse_options),
                convert_options=convert_options,
            )
        except pa.ArrowInvalid as e:
            raise ParserError(e) from e

        dtype_backend = self.kwds["dtype_backend"]

        # Convert all pa.null() cols -> float64 (non nullable)
        # else Int64 (nullable case, see below)
        if dtype_backend is lib.no_default:
            new_schema = table.schema
            new_type = pa.float64()
            for i, arrow_type in enumerate(table.schema.types):
                if pa.types.is_null(arrow_type):
                    new_schema = new_schema.set(
                        i, new_schema.field(i).with_type(new_type)
                    )

            table = table.cast(new_schema)

        multi_index_named = self._adjust_column_names(table)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "make_block is deprecated",
                Pandas4Warning,
            )
            frame = arrow_table_to_pandas(
                table,
                dtype_backend=dtype_backend,
                null_to_int64=True,
                dtype=self.dtype,
                names=self.names,
            )

        if self.header is None:
            frame.columns = self.names

        return self._finalize_pandas_output(frame, multi_index_named)
