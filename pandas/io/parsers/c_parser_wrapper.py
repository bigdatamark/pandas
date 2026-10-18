# [MODULE DOC]
# Adapter that lets the delimited-text reader use the compiled C tokenizer
# (the text reader in the compiled parsers module) as an engine. Public
# content: the CParserWrapper class (a parser base subclass created by the
# readers module for the C engine) plus helpers to filter column selections,
# concatenate low-memory chunks (warning on mixed types) and normalize dtype
# arguments to dtype objects. Also re-exports names imported from the base
# parser module for use by other modules and tests. No import-time side
# effects beyond imports.
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs import (
    lib,
    parsers,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DtypeWarning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.concat import (
    concat_compat,
    union_categoricals,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core.indexes.api import ensure_index_from_sequences

from pandas.io.common import (
    dedup_names,
    is_potential_multi_index,
)
from pandas.io.parsers.base_parser import (
    ParserBase,
    ParserError,
    date_converter,
    evaluate_callable_usecols,
    is_index_col,
    validate_parse_dates_presence,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        DtypeArg,
        DtypeObj,
        ReadCsvBuffer,
        SequenceT,
    )

    from pandas import (
        Index,
        MultiIndex,
    )


class CParserWrapper(ParserBase):
    # [CLASS DOC]
    # Parsing engine wrapping the compiled C text reader. Extends the shared
    # parser base. State: low_memory (read in chunks and concatenate), the
    # underlying reader object, unnamed_cols (placeholder names found in the
    # header), names/original names/index names resolved from the header and
    # options, and the implicit-index flag (true when the data has more fields
    # than the header, so leading columns form the index). Key methods: close,
    # read (returns index, column labels and column data for up to a number of
    # rows; repeated calls stream chunks, raising end-of-iteration when
    # exhausted) and helpers to mark date columns as unconverted.
    low_memory: bool
    _reader: parsers.TextReader

    def __init__(self, src: ReadCsvBuffer[str], **kwds) -> None:
        # [FUNCTION DOC]
        # Builds the C reader and resolves column names from the header.
        # Inputs: src a text buffer; kwds the option mapping (stored; a copy
        # is used for the reader). The copy gets: low-memory removed and
        # stored (default false); a leading-columns allowance unless the index
        # option is exactly false; the validated column selection; the bad-lines
        # policy as its integer value; storage, encoding, memory-map and
        # compression keys dropped (handled earlier); the dtype argument
        # normalized to dtype objects; and the dtype backend set to numpy when
        # absent or unset (the pyarrow backend requires pyarrow, checked
        # eagerly so failure happens before reading).
        # After constructing the reader: records its unnamed-column set; sets
        # names to none if the reader found no header, else derives names, index
        # names, column-level names and a passed-names flag from the header rows
        # (multi-row headers become tuples); if still no names, uses numeric
        # positions up to the table width. Saves a copy of the names as original
        # names, then, when a column selection exists, resolves callables to
        # positions, validates string selections against the names (value error
        # for missing columns) and filters names down to the selected ones,
        # validating again if fewer names than selections remain. Validates
        # date columns exist, marks date columns as exempt from conversion in the
        # reader, then sets original names to the filtered names (no copy).
        # If the reader reports no leading columns and an index option is set,
        # cleans index names/positions from the names. With no header and no
        # user-passed names, index names are reset to none placeholders. Sets
        # the implicit-index flag when the reader reports leading columns.
        # Raises: value errors from validation; import error for the missing
        # pyarrow dependency.
        super().__init__(kwds)
        self.kwds = kwds
        kwds = kwds.copy()

        self.low_memory = kwds.pop("low_memory", False)

        # #2442
        kwds["allow_leading_cols"] = self.index_col is not False

        # GH20529, validate usecol arg before TextReader
        kwds["usecols"] = self.usecols

        # Have to pass int, would break tests using TextReader directly otherwise :(
        kwds["on_bad_lines"] = self.on_bad_lines.value

        for key in (
            "storage_options",
            "encoding",
            "memory_map",
            "compression",
        ):
            kwds.pop(key, None)

        kwds["dtype"] = ensure_dtype_objs(kwds.get("dtype", None))
        if "dtype_backend" not in kwds or kwds["dtype_backend"] is lib.no_default:
            kwds["dtype_backend"] = "numpy"
        if kwds["dtype_backend"] == "pyarrow":
            # Fail here loudly instead of in cython after reading
            import_optional_dependency("pyarrow")
        self._reader = parsers.TextReader(src, **kwds)

        self.unnamed_cols = self._reader.unnamed_cols

        passed_names = self.names is None

        if self._reader.header is None:
            self.names = None
        else:
            (
                self.names,
                self.index_names,
                self.col_names,
                passed_names,
            ) = self._extract_multi_indexer_columns(
                self._reader.header,
                self.index_names,
                passed_names,
            )

        if self.names is None:
            self.names = list(range(self._reader.table_width))

        # gh-9755
        #
        # need to set orig_names here first
        # so that proper indexing can be done
        # with _set_noconvert_columns
        #
        # once names has been filtered, we will
        # then set orig_names again to names
        self.orig_names = self.names[:]

        if self.usecols:
            usecols = evaluate_callable_usecols(self.usecols, self.orig_names)

            # GH 14671
            # assert for mypy, orig_names is List or None, None would error in issubset
            assert self.orig_names is not None
            if self.usecols_dtype == "string" and not set(usecols).issubset(
                self.orig_names
            ):
                self._validate_usecols_names(usecols, self.orig_names)

            if len(self.names) > len(usecols):
                self.names = [
                    n
                    for i, n in enumerate(self.names)
                    if (i in usecols or n in usecols)
                ]

            if len(self.names) < len(usecols):
                self._validate_usecols_names(
                    usecols,
                    self.names,
                )

        validate_parse_dates_presence(self.parse_dates, self.names)
        self._set_noconvert_columns()

        self.orig_names = self.names

        if self._reader.leading_cols == 0 and is_index_col(self.index_col):
            (
                index_names,
                self.names,
                self.index_col,
            ) = self._clean_index_names(
                self.names,
                self.index_col,
            )

            if self.index_names is None:
                self.index_names = index_names

        if self._reader.header is None and not passed_names:
            assert self.index_names is not None
            self.index_names = [None] * len(self.index_names)

        self._implicit_index = self._reader.leading_cols > 0

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the underlying reader and any handles it opened. A value error
        # raised by the reader (already closed) is swallowed. Returns nothing.
        # close handles opened by C parser
        try:
            self._reader.close()
        except ValueError:
            pass

    def _set_noconvert_columns(self) -> None:
        # [FUNCTION DOC]
        # HELPER: tells the C reader which columns must not be dtype-converted
        # (date columns). Maps current names to their positions in the original
        # names, asks the base class helper for the exempt positions, and
        # marks each in the reader (side effect on the reader). Requires
        # original names to be set. Returns nothing.
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
        assert self.orig_names is not None
        # error: Cannot determine type of 'names'

        # much faster than using orig_names.index(x) xref GH#44106
        names_dict = {x: i for i, x in enumerate(self.orig_names)}
        col_indices = [names_dict[x] for x in self.names]
        noconvert_columns = self._set_noconvert_dtype_columns(
            col_indices,
            self.names,
        )
        for col in noconvert_columns:
            self._reader.set_noconvert(col)

    def _low_memory_column_labels(
        self, chunks: list[dict[int, ArrayLike]]
    ) -> dict[int, Hashable]:
        # [FUNCTION DOC]
        # HELPER: for the low-memory read path, builds a mapping from the
        # integer field positions that key the chunk dictionaries to column
        # labels, used for the mixed-type warning. Takes the list of chunks and
        # uses the sorted positions of the first chunk; positions after any
        # leading implicit index columns are paired with original names in order
        # (pairing stops at the shorter side, non-strict); positions without a
        # name (including leading index columns) map to none. Returns a new
        # dictionary covering all positions.
        """
        Map the field positions ``read_low_memory`` keys its chunks by to labels.

        The positions are those of the source row, so they skip whatever
        ``usecols`` dropped and start past a leading implicit index column.
        ``orig_names`` holds exactly the named columns that remain, in the same
        order, which is how ``read`` renames these keys further down.  A leading
        implicit index column has no name of its own and maps to ``None``.
        """
        assert self.orig_names is not None
        positions = sorted(chunks[0])
        named = dict(
            zip(
                positions[self._reader.leading_cols :],
                self.orig_names,
                strict=False,
            )
        )
        return {position: named.get(position) for position in positions}

    def read(
        self,
        nrows: int | None = None,
    ) -> tuple[
        Index | MultiIndex | None,
        Sequence[Hashable] | MultiIndex,
        Mapping[Hashable, AnyArrayLike],
    ]:
        # [FUNCTION DOC]
        # Reads up to nrows rows (all remaining when none) and returns
        # (index or none, column labels, mapping of label to column array).
        # Reading: in low-memory mode reads chunks and concatenates them per
        # column (destroying the chunks; may warn about mixed types);
        # otherwise a single read. If the reader signals exhaustion: on the
        # first read, returns an empty result built from the header names (with
        # deduplicated names, empty index and series with the dtype option,
        # column labels turned into a multi-level index if needed and
        # restricted to the column selection); on later reads closes the reader
        # and re-raises the end-of-iteration signal. Marks the first chunk done.
        # With leading implicit-index columns: raises a parser error if the
        # index option's length differs from their count; pops those columns
        # from the data, date-converts them where requested, and builds the
        # index from them; then filters and deduplicates names, relabels the
        # remaining columns by sorted position, applies date conversions and
        # builds column labels (possibly multi-level). Otherwise: relabels
        # columns by sorted position using deduplicated, filtered original
        # names, checks data length against names when no column selection
        # exists (may warn), applies date conversions and builds index and
        # labels from the index option.
        # KNOWN ISSUE: in the implicit-index branch the relabeling pairs names
        # with data strictly, raising a value error on count mismatch, while the
        # other branch pairs non-strictly and silently truncates.
        index: Index | MultiIndex | None
        column_names: Sequence[Hashable] | MultiIndex
        try:
            if self.low_memory:
                chunks = self._reader.read_low_memory(nrows)
                # destructive to chunks
                data = _concatenate_chunks(
                    chunks, self._low_memory_column_labels(chunks)
                )
            else:
                data = self._reader.read(nrows)
        except StopIteration:
            if self._first_chunk:
                self._first_chunk = False
                # assert for mypy, orig_names is List or None, None would error in
                # list(...) in dedup_names
                assert self.orig_names is not None
                names = dedup_names(
                    self.orig_names,
                    is_potential_multi_index(self.orig_names, self.index_col),
                )
                index, columns, col_dict = self._get_empty_meta(
                    names,
                    dtype=self.dtype,
                )
                # error: Incompatible types in assignment (expression has type
                # "list[Hashable] | MultiIndex", variable has type "list[Hashable]")
                columns = self._maybe_make_multi_index_columns(  # type: ignore[assignment]
                    columns, self.col_names
                )

                columns = _filter_usecols(self.usecols, columns)
                columns_set = set(columns)

                col_dict = {k: v for k, v in col_dict.items() if k in columns_set}

                return index, columns, col_dict

            else:
                self.close()
                raise

        # Done with first read, next time raise StopIteration
        self._first_chunk = False

        names = self.names

        if self._reader.leading_cols:
            # implicit index, no index names
            arrays = []

            if self.index_col and self._reader.leading_cols != len(self.index_col):
                raise ParserError(
                    "Could not construct index. Requested to use "
                    f"{len(self.index_col)} number of columns, but "
                    f"{self._reader.leading_cols} left to parse."
                )

            for i in range(self._reader.leading_cols):
                if self.index_col is None:
                    values = data.pop(i)
                else:
                    values = data.pop(self.index_col[i])

                if self._should_parse_dates(i):
                    values = date_converter(
                        values,
                        col=(
                            self.index_names[i]
                            if self.index_names is not None
                            else None
                        ),
                        dayfirst=self.dayfirst,
                        cache_dates=self.cache_dates,
                        date_format=self.date_format,
                    )
                arrays.append(values)

            index = ensure_index_from_sequences(arrays)

            names = _filter_usecols(self.usecols, names)

            names = dedup_names(names, is_potential_multi_index(names, self.index_col))

            # rename dict keys
            data_tups = sorted(data.items())
            data = {k: v for k, (i, v) in zip(names, data_tups, strict=True)}

            date_data = self._do_date_conversions(names, data)

            # maybe create a mi on the columns
            column_names = self._maybe_make_multi_index_columns(names, self.col_names)

        else:
            # rename dict keys
            data_tups = sorted(data.items())

            # ugh, mutation

            # assert for mypy, orig_names is List or None, None would error in list(...)
            assert self.orig_names is not None
            names = list(self.orig_names)
            names = dedup_names(names, is_potential_multi_index(names, self.index_col))

            names = _filter_usecols(self.usecols, names)

            # columns as list
            alldata = [x[1] for x in data_tups]
            if self.usecols is None:
                self._check_data_length(names, alldata)

            data = {k: v for k, (i, v) in zip(names, data_tups, strict=False)}

            date_data = self._do_date_conversions(names, data)
            index, column_names = self._make_index(alldata, names)

        return index, column_names, date_data


def _filter_usecols(usecols, names: SequenceT) -> SequenceT | list[Hashable]:
    # [FUNCTION DOC]
    # HELPER: restricts column labels to the selected columns. Resolves a
    # callable selection to positions first. If a selection exists and its size
    # differs from the label count, returns a new list of labels whose position
    # or value is in the selection; otherwise returns the labels unchanged.
    # hackish
    usecols = evaluate_callable_usecols(usecols, names)
    if usecols is not None and len(names) != len(usecols):
        return [name for i, name in enumerate(names) if i in usecols or name in usecols]
    return names


def _concatenate_chunks(
    chunks: list[dict[int, ArrayLike]],
    column_names: Sequence[Hashable] | Mapping[int, Hashable],
) -> dict:
    # [FUNCTION DOC]
    # HELPER: merges the per-chunk column dictionaries from a low-memory read.
    # Inputs: chunks list of mappings from column key to array (consumed:
    # entries are popped out); column_names indexable by those keys, used for
    # the warning text. For each key of the first chunk, collects its arrays;
    # categorical columns are merged with a categorical union without sorting
    # categories; others are concatenated. If more than one non-categorical
    # dtype appeared and the result is object dtype, the column is recorded and
    # one dtype warning is emitted at the end listing position and label (label
    # omitted when none) and advising to set dtype or disable low-memory mode.
    # Returns a dictionary of key to merged array.
    # KNOWN ISSUE: the warning text is assembled by joining a one-element list,
    # which is redundant but harmless.
    """
    Concatenate chunks of data read with low_memory=True.

    The tricky part is handling Categoricals, where different chunks
    may have different inferred categories.

    ``column_names`` names the keys of ``chunks`` for the mixed-dtype warning,
    so it has to be indexable by whatever those keys are: the low_memory reader
    keys its chunks by field position and passes a mapping keyed the same way,
    while the parallel reader keys its own by name.
    """
    names = list(chunks[0].keys())
    warning_columns = []

    result: dict = {}
    for name in names:
        arrs = [chunk.pop(name) for chunk in chunks]
        # Check each arr for consistent types.
        dtypes = {a.dtype for a in arrs}
        non_cat_dtypes = {x for x in dtypes if not isinstance(x, CategoricalDtype)}

        dtype = dtypes.pop()
        if isinstance(dtype, CategoricalDtype):
            result[name] = union_categoricals(arrs, sort_categories=False)
        else:
            result[name] = concat_compat(arrs)
            if len(non_cat_dtypes) > 1 and result[name].dtype == np.dtype(object):
                warning_columns.append((name, column_names[name]))

    if warning_columns:
        warning_names = ", ".join(
            [
                f"{position}: {label}" if label is not None else f"{position}"
                for position, label in warning_columns
            ]
        )
        warning_message = " ".join(
            [
                f"Columns ({warning_names}) have mixed types. "
                f"Specify dtype option on import or set low_memory=False."
            ]
        )
        warnings.warn(warning_message, DtypeWarning, stacklevel=find_stack_level())
    return result


def ensure_dtype_objs(
    dtype: DtypeArg | dict[Hashable, DtypeArg] | None,
) -> DtypeObj | dict[Hashable, DtypeObj] | None:
    # [FUNCTION DOC]
    # Normalizes a dtype argument to dtype objects. None stays None; a
    # default-valued mapping becomes a new default-valued mapping whose default
    # and entries are converted (the default factory's result is converted
    # once); a plain mapping becomes a new mapping with converted values; any
    # other value is converted to a single dtype object. Raises whatever dtype
    # conversion raises for invalid dtypes.
    """
    Ensure we have either None, a dtype object, or a dictionary mapping to
    dtype objects.
    """
    if isinstance(dtype, defaultdict):
        # "None" not callable  [misc]
        default_dtype = pandas_dtype(dtype.default_factory())  # type: ignore[misc]
        dtype_converted: defaultdict = defaultdict(lambda: default_dtype)
        for key in dtype.keys():
            dtype_converted[key] = pandas_dtype(dtype[key])
        return dtype_converted
    elif isinstance(dtype, dict):
        return {k: pandas_dtype(dtype[k]) for k in dtype}
    elif dtype is not None:
        return pandas_dtype(dtype)
    return dtype
