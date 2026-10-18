# [MODULE DOC]
# Builds the text summary shown by the info methods of frames and series (class line,
#   index range, columns table, dtype counts and memory usage).
# Public contents: DataFrameInfo and SeriesInfo (data wrappers exposing dtypes, non-null
#   counts, dtype counts and memory usage, each with a render method), and module level
#   docstring fragments shared by the info methods: the show counts description, the
#   series examples, the series see-also and max columns texts, and the dictionary of
#   keyword substitutions used to fill the series info docstring template.
# Private structure: a base info class; a printer layer (one printer for frames and one
#   for series) that decides which table builder to use from verbosity and display
#   options; a builder layer (abstract builder, frame builders in verbose and non
#   verbose forms, series builders in verbose and non verbose forms, and a mixin holding
#   the shared logic of verbose tables: column widths, header, separator and body
#   lines); and small helpers for padding strings, formatting sizes and counting dtypes.
# Relationships: output is written with the buffer writing helper of the format module;
#   values are rendered with the pretty printer; options for display limits and memory
#   usage are read from the global options.
# Import-time side effects: builds the docstring fragment strings and substitution
#   dictionary.

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

from pandas._config import get_option

from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        Dtype,
        WriteBuffer,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )

show_counts_sub = dedent(
    """\
    show_counts : bool, optional
        Whether to show the non-null counts. By default, this is shown
        only if the DataFrame is smaller than
        ``pandas.options.display.max_info_rows`` and
        ``pandas.options.display.max_info_columns``. A value of True always
        shows the counts, and False never shows the counts."""
)

series_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> s = pd.Series(text_values, index=int_values)
    >>> s.info()
    <class 'pandas.Series'>
    Index: 5 entries, 1 to 5
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    5 non-null      object
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Prints a summary excluding information about its values:

    >>> s.info(verbose=False)
    <class 'pandas.Series'>
    Index: 5 entries, 1 to 5
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Pipe output of Series.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> s.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big Series and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> s = pd.Series(np.random.choice(['a', 'b', 'c'], 10 ** 6))
    >>> s.info()
    <class 'pandas.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 7.6+ MB

    >>> s.info(memory_usage='deep')
    <class 'pandas.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 55.3 MB"""
)


series_see_also_sub = dedent(
    """\
    Series.describe: Generate descriptive statistics of Series.
    Series.memory_usage: Memory usage of Series."""
)
series_max_cols_sub = dedent(
    """\
    max_cols : int, optional
        Unused, exists only for compatibility with DataFrame.info."""
)


series_sub_kwargs = {
    "klass": "Series",
    "type_sub": "",
    "max_cols_sub": series_max_cols_sub,
    "show_counts_sub": show_counts_sub,
    "examples_sub": series_examples_sub,
    "see_also_sub": series_see_also_sub,
    "version_added_sub": "\n.. versionadded:: 1.4.0\n",
}


def _put_str(s: str | Dtype, space: int) -> str:
    # [FUNCTION DOC]
    # HELPER: pads or truncates text to exactly the given width. Input: s (anything,
    #   converted to text) and space (target width). Returns the text cut to at most
    #   that many characters and left justified (padded with spaces on the right).
    """
    Make string of specified length, padding to the right if necessary.

    Parameters
    ----------
    s : Union[str, Dtype]
        String to be formatted.
    space : int
        Length to force string to be of.

    Returns
    -------
    str
        String coerced to given length.

    Examples
    --------
    >>> pd.io.formats.info._put_str("panda", 6)
    'panda '
    >>> pd.io.formats.info._put_str("panda", 4)
    'pand'
    """
    return str(s)[:space].ljust(space)


def _sizeof_fmt(num: float, size_qualifier: str) -> str:
    # [FUNCTION DOC]
    # HELPER: formats a byte count in human readable form. Inputs: num (size in bytes)
    #   and size_qualifier (empty text or a plus sign for a lower bound). Divides by
    #   1024 repeatedly through bytes, KB, MB, GB and TB and returns the number with one
    #   decimal place, the qualifier and the unit; larger values are expressed in PB.
    """
    Return size in human readable format.

    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).

    Returns
    -------
    str
        Size in human readable format.

    Examples
    --------
    >>> _sizeof_fmt(23028, "")
    '22.5 KB'

    >>> _sizeof_fmt(23028, "+")
    '22.5+ KB'
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f}{size_qualifier} {x}"
        num /= 1024.0
    return f"{num:3.1f}{size_qualifier} PB"


def _initialize_memory_usage(
    memory_usage: bool | str | None = None,
) -> bool | str:
    # [FUNCTION DOC]
    # HELPER: resolves the memory usage setting. Input: memory_usage, a boolean, the
    #   string deep, or none. If none, returns the value of the memory usage display
    #   option; otherwise returns the input unchanged.
    """Get memory usage based on inputs and display options."""
    if memory_usage is None:
        memory_usage = get_option("display.memory_usage")
    return memory_usage


class _BaseInfo(ABC):
    # [CLASS DOC]
    # HELPER: abstract base class for the info data wrappers (DataFrameInfo and
    #   SeriesInfo). Declared attributes: data (the frame or series) and memory_usage
    #   (false, true or the string deep, where deep means introspect object dtypes).
    #   Abstract members the subclasses supply: dtypes, dtype_counts, non_null_counts,
    #   memory_usage_bytes and render. Provides the human readable memory usage text and
    #   the size qualifier logic.
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    """

    data: DataFrame | Series
    memory_usage: bool | str

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        # [FUNCTION DOC]
        # Abstract property: the dtype of each column (a single one for a series).
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        # [FUNCTION DOC]
        # Abstract property: mapping from dtype name to the number of columns having it.
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> list[int] | Series:
        # [FUNCTION DOC]
        # Abstract property: non null counts for every column (a one element sequence
        #   for a series).
        """Sequence of non-null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        # [FUNCTION DOC]
        # Abstract property: total memory usage of the data in bytes.
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """

    @property
    def memory_usage_string(self) -> str:
        # [FUNCTION DOC]
        # Read only property: memory usage as a human readable text (using the size
        #   qualifier) followed by a newline.
        """Memory usage in a form of human readable string."""
        return f"{_sizeof_fmt(self.memory_usage_bytes, self.size_qualifier)}\n"

    @property
    def size_qualifier(self) -> str:
        # [FUNCTION DOC]
        # Read only property: returns a plus sign when memory usage is requested but not
        #   in deep mode and the usage is only a lower bound, which is the case when any
        #   object dtype is present or the index reports that its memory usage is
        #   qualified; otherwise returns the empty string. The check is best effort and
        #   misses some cases such as categorical data with object categories.
        size_qualifier = ""
        if self.memory_usage:
            if self.memory_usage != "deep":
                # size_qualifier is just a best effort; not guaranteed to catch
                # all cases (e.g., it misses categorical data even with object
                # categories)
                if (
                    "object" in self.dtype_counts
                    or self.data.index._is_memory_usage_qualified
                ):
                    size_qualifier = "+"
        return size_qualifier

    @abstractmethod
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        # [FUNCTION DOC]
        # Abstract method: writes the info report. Keyword only inputs: buf (writable
        #   text buffer or none for standard output), max_cols (column limit for verbose
        #   output), verbose (force or suppress the detailed table) and show_counts
        #   (force or suppress non null counts). Returns nothing.
        pass


class DataFrameInfo(_BaseInfo):
    # [CLASS DOC]
    # Concrete info wrapper for frames. State: data (the frame) and memory_usage
    #   (resolved memory usage setting). Provides dtypes, column labels, column count,
    #   non null counts per column, dtype counts, memory usage in bytes, and render
    #   which prints through the frame printer.
    """
    Class storing dataframe-specific info.
    """

    def __init__(
        self,
        data: DataFrame,
        memory_usage: bool | str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the frame and resolve the memory usage setting (none means use the
        #   display option).
        self.data: DataFrame = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        # [FUNCTION DOC]
        # Read only property: mapping from dtype name to number of columns, computed
        #   with the shared dtype counting helper.
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        # [FUNCTION DOC]
        # Read only property: the dtypes of the frame's columns.
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        return self.data.dtypes

    @property
    def ids(self) -> Index:
        # [FUNCTION DOC]
        # Read only property: the frame's column labels.
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        return self.data.columns

    @property
    def col_count(self) -> int:
        # [FUNCTION DOC]
        # Read only property: the number of columns to summarize.
        """Number of columns to be summarized."""
        return len(self.ids)

    @property
    def non_null_counts(self) -> Series:
        # [FUNCTION DOC]
        # Read only property: the number of non null values in each column, as a series.
        """Sequence of non-null counts for all columns or column (if series)."""
        return self.data.count()

    @property
    def memory_usage_bytes(self) -> int:
        # [FUNCTION DOC]
        # Read only property: total memory in bytes of the frame including the index;
        #   the object dtypes are deeply introspected when memory usage is set to deep.
        #   Returns the sum as a numeric scalar.
        deep = self.memory_usage == "deep"
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        # [FUNCTION DOC]
        # Print the report. Keyword only inputs: buf, max_cols, verbose, show_counts
        #   (all required, see the abstract method). Creates a frame printer with these
        #   options and writes to the buffer. Returns nothing.
        printer = _DataFrameInfoPrinter(
            info=self,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)


class SeriesInfo(_BaseInfo):
    # [CLASS DOC]
    # Concrete info wrapper for series. State: data (the series) and memory_usage
    #   (resolved setting). Provides one element sequences for dtypes and non null
    #   counts, dtype counts via a one column frame, memory usage in bytes, and render.
    """
    Class storing series-specific info.
    """

    def __init__(
        self,
        data: Series,
        memory_usage: bool | str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the series and resolve the memory usage setting (none means use the
        #   display option).
        self.data: Series = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None = None,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Print the report for a series. Keyword only inputs with defaults of none: buf,
        #   max_cols, verbose, show_counts. Raises a value error if max_cols is given,
        #   since it is only meaningful for frames. Creates a series printer and writes
        #   to the buffer. Returns nothing.
        if max_cols is not None:
            raise ValueError(
                "Argument `max_cols` can only be passed "
                "in DataFrame.info, not Series.info"
            )
        printer = _SeriesInfoPrinter(
            info=self,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)

    @property
    def non_null_counts(self) -> list[int]:
        # [FUNCTION DOC]
        # Read only property: a one element list holding the number of non null values.
        return [self.data.count()]

    @property
    def dtypes(self) -> Iterable[Dtype]:
        # [FUNCTION DOC]
        # Read only property: a one element list holding the series dtype.
        return [self.data.dtypes]

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        # [FUNCTION DOC]
        # Read only property: mapping from dtype name to count (always one entry),
        #   computed by wrapping the series in a one column frame.
        from pandas.core.frame import DataFrame

        return _get_dataframe_dtype_counts(DataFrame(self.data))

    @property
    def memory_usage_bytes(self) -> int:
        # [FUNCTION DOC]
        # Read only property: memory usage of the series in bytes including the index;
        #   object data is deeply introspected when memory usage is set to deep.
        """Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """
        deep = self.memory_usage == "deep"
        return self.data.memory_usage(index=True, deep=deep)


class _InfoPrinterAbstract:
    # [CLASS DOC]
    # HELPER: common base of the printer classes. Subclasses must create a table
    #   builder; the base class turns the builder's lines into output.
    # KNOWN ISSUE: the create-builder method is marked abstract, but the class does not
    #   derive from the abstract base class helper, so the abstractness is not enforced.
    """
    Class for printing dataframe or series info.
    """

    def to_buffer(self, buf: WriteBuffer[str] | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: writes the info text. Input: buf, a writable text buffer or none (none
        #   means standard output). Asks the builder for the list of lines and writes
        #   them to the buffer with the shared buffer writing helper. Returns nothing.
        """Save dataframe info into buffer."""
        table_builder = self._create_table_builder()
        lines = table_builder.get_lines()
        if buf is None:  # pragma: no cover
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> _TableBuilderAbstract:
        # [FUNCTION DOC]
        # HELPER: abstract hook that creates the table builder appropriate to the data.
        #   Returns the builder.
        """Create instance of table builder."""


class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    # [CLASS DOC]
    # HELPER: printer for frame info. State: info (the DataFrameInfo), data (the frame),
    #   verbose (true, false or none), max_cols (limit of columns before switching to
    #   compact output; defaults from the display option) and show_counts (whether to
    #   show non null counts; if not given, true only when neither the column nor the
    #   row limits are exceeded). Chooses between the verbose and non verbose builder.
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: DataFrameInfo,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the info object, its frame and the verbosity; resolve max_cols from the
        #   display option when none; resolve show_counts from the limits when none.
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.max_cols = self._initialize_max_cols(max_cols)
        self.show_counts = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self) -> int:
        # [FUNCTION DOC]
        # Read only property: the display option for the maximum number of rows for
        #   which counts are shown.
        """Maximum info rows to be displayed."""
        return get_option("display.max_info_rows")

    @property
    def exceeds_info_cols(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: true when the number of columns is greater than max_cols.
        # KNOWN ISSUE: the docstring says it checks that the column count does not
        #   exceed the maximum, which is the opposite of what is computed (true means
        #   the maximum is exceeded).
        """Check if number of columns to be summarized does not exceed maximum."""
        return bool(self.col_count > self.max_cols)

    @property
    def exceeds_info_rows(self) -> bool:
        # [FUNCTION DOC]
        # Read only property: true when the number of rows is greater than the maximum
        #   info rows option.
        # KNOWN ISSUE: the docstring says it checks that the row count does not exceed
        #   the maximum, which is the opposite of what is computed (true means the
        #   maximum is exceeded).
        """Check if number of rows to be summarized does not exceed maximum."""
        return bool(len(self.data) > self.max_rows)

    @property
    def col_count(self) -> int:
        # [FUNCTION DOC]
        # Read only property: number of columns, taken from the info object.
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_max_cols(self, max_cols: int | None) -> int:
        # [FUNCTION DOC]
        # HELPER: returns the given column limit, or the maximum info columns display
        #   option when none.
        if max_cols is None:
            return get_option("display.max_info_columns")
        return max_cols

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns the given flag, or, when none, true only if neither the column
        #   limit nor the row limit is exceeded.
        if show_counts is None:
            return bool(not self.exceeds_info_cols and not self.exceeds_info_rows)
        else:
            return show_counts

    def _create_table_builder(self) -> _DataFrameTableBuilder:
        # [FUNCTION DOC]
        # HELPER: picks the frame table builder. A truthy verbose chooses the verbose
        #   builder (with counts as configured); an explicit false chooses the non
        #   verbose builder; when verbose is none, chooses the non verbose builder if
        #   the column limit is exceeded, else the verbose builder.
        """
        Create instance of table builder based on verbosity and display settings.
        """
        if self.verbose:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        elif self.verbose is False:  # specifically set to False, not necessarily None
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        elif self.exceeds_info_cols:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        else:
            return _DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )


class _SeriesInfoPrinter(_InfoPrinterAbstract):
    # [CLASS DOC]
    # HELPER: printer for series info. State: info (the SeriesInfo), data, verbose and
    #   show_counts (true unless given). Chooses between the verbose and non verbose
    #   series builder.
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesInfo
        Instance of SeriesInfo.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: SeriesInfo,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the info object, its series and the verbosity; resolve show_counts
        #   (default true).
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.show_counts = self._initialize_show_counts(show_counts)

    def _create_table_builder(self) -> _SeriesTableBuilder:
        # [FUNCTION DOC]
        # HELPER: picks the series table builder: verbose when verbose is true or none,
        #   otherwise non verbose.
        """
        Create instance of table builder based on verbosity.
        """
        if self.verbose or self.verbose is None:
            return _SeriesTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        else:
            return _SeriesTableBuilderNonVerbose(info=self.info)

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        # [FUNCTION DOC]
        # HELPER: returns the given flag, or true when none.
        if show_counts is None:
            return True
        else:
            return show_counts


class _TableBuilderAbstract(ABC):
    # [CLASS DOC]
    # HELPER: abstract base of the table builders. Declares the accumulated list of
    #   output lines and the info object. Offers properties delegating to the info
    #   object (data, dtypes, dtype counts, whether to display memory usage, memory
    #   usage text, non null counts) and methods that add the shared lines (object type,
    #   index range, dtype summary).
    """
    Abstract builder for info table.
    """

    _lines: list[str]
    info: _BaseInfo

    @abstractmethod
    def get_lines(self) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: abstract method returning the list of text lines of the report.
        """Product in a form of list of lines (strings)."""

    @property
    def data(self) -> DataFrame | Series:
        # [FUNCTION DOC]
        # HELPER: property returning the data of the info object.
        return self.info.data

    @property
    def dtypes(self) -> Iterable[Dtype]:
        # [FUNCTION DOC]
        # HELPER: property returning the dtypes from the info object.
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        # [FUNCTION DOC]
        # HELPER: property returning the dtype to count mapping from the info object.
        """Mapping dtype - number of counts."""
        return self.info.dtype_counts

    @property
    def display_memory_usage(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property that is true if memory usage reporting is enabled (any truthy
        #   setting).
        """Whether to display memory usage."""
        return bool(self.info.memory_usage)

    @property
    def memory_usage_string(self) -> str:
        # [FUNCTION DOC]
        # HELPER: property returning the memory usage text with the qualifier, from the
        #   info object.
        """Memory usage string with proper size qualifier."""
        return self.info.memory_usage_string

    @property
    def non_null_counts(self) -> list[int] | Series:
        # [FUNCTION DOC]
        # HELPER: property returning the non null counts from the info object.
        return self.info.non_null_counts

    def add_object_type_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends a line holding the string form of the data's type.
        """Add line with string representation of dataframe to the table."""
        self._lines.append(str(type(self.data)))

    def add_index_range_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends the summary line of the data's index (for example its range
        #   and length).
        """Add line with range of indices to the table."""
        self._lines.append(self.data.index._summary())

    def add_dtypes_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends a line listing the dtypes present, sorted by name, each as the
        #   name followed by the count in parentheses and separated by commas, prefixed
        #   with dtypes.
        """Add summary line with dtypes present in dataframe."""
        collected_dtypes = [
            f"{key}({val:d})" for key, val in sorted(self.dtype_counts.items())
        ]
        self._lines.append(f"dtypes: {', '.join(collected_dtypes)}")


class _DataFrameTableBuilder(_TableBuilderAbstract):
    # [CLASS DOC]
    # HELPER: abstract builder for the frame report. Holds the DataFrameInfo, produces
    #   the empty frame report when there are no columns, and delegates the non empty
    #   case to subclasses. Provides the frame, its column labels and column count, and
    #   the memory usage line.
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.
    """

    def __init__(self, *, info: DataFrameInfo) -> None:
        # [FUNCTION DOC]
        # Keyword only constructor taking the DataFrameInfo, which is stored.
        self.info: DataFrameInfo = info

    def get_lines(self) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: builds the report. Resets the list of lines, fills either the empty
        #   frame report (no columns) or the non empty report, and returns the lines.
        self._lines = []
        if self.col_count == 0:
            self._fill_empty_info()
        else:
            self._fill_non_empty_info()
        return self._lines

    def _fill_empty_info(self) -> None:
        # [FUNCTION DOC]
        # HELPER: for a frame without columns appends the object type line, the index
        #   range line and a line stating the frame type is empty.
        """Add lines to the info table, pertaining to empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self._lines.append(f"Empty {type(self.data).__name__}\n")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        # [FUNCTION DOC]
        # HELPER: abstract method that appends the lines for a frame with columns.
        """Add lines to the info table, pertaining to non-empty dataframe."""

    @property
    def data(self) -> DataFrame:
        # [FUNCTION DOC]
        # HELPER: property returning the frame.
        """DataFrame."""
        return self.info.data

    @property
    def ids(self) -> Index:
        # [FUNCTION DOC]
        # HELPER: property returning the column labels.
        """Dataframe columns."""
        return self.info.ids

    @property
    def col_count(self) -> int:
        # [FUNCTION DOC]
        # HELPER: property returning the number of columns.
        """Number of dataframe columns to be summarized."""
        return self.info.col_count

    def add_memory_usage_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends a line with the memory usage text.
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")


class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    # [CLASS DOC]
    # HELPER: frame builder for compact output without the per column table. The report
    #   has the object type, index range, a summary of the column labels, the dtype
    #   summary and, if enabled, memory usage.
    """
    Dataframe info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends, in order, the object type line, index range line, columns
        #   summary line, dtypes line and (if enabled) the memory usage line.
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_columns_summary_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends the summary line of the column labels, labeled as columns.
        self._lines.append(self.ids._summary(name="Columns"))


class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    # [CLASS DOC]
    # HELPER: mixin with the shared logic of verbose tables, used by the frame and
    #   series verbose builders. Declares the spacing between columns (two spaces), the
    #   table body rows as strings, the final column widths and whether non null counts
    #   are shown. Requires subclasses to provide headers and the generators of body
    #   rows with and without counts.
    """
    Mixin for verbose info output.
    """

    SPACING: str = " " * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        # [FUNCTION DOC]
        # HELPER: abstract property: the column titles of the verbose table.
        """Headers names of the columns in verbose table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        # [FUNCTION DOC]
        # HELPER: property returning the length of each header title.
        """Widths of header columns (only titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        # [FUNCTION DOC]
        # HELPER: returns, for every column, the larger of the header width and the
        #   widest body cell.
        """Get widths of columns containing both headers and actual content."""
        body_column_widths = self._get_body_column_widths()
        return [
            max(*widths)
            for widths in zip(
                self.header_column_widths, body_column_widths, strict=False
            )
        ]

    def _get_body_column_widths(self) -> Sequence[int]:
        # [FUNCTION DOC]
        # HELPER: returns the widest cell of each column of the body rows.
        """Get widths of table content columns."""
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows, strict=True))
        return [max(len(x) for x in col) for col in strcols]

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        # [FUNCTION DOC]
        # HELPER: returns the iterator of body rows, with the non null count column when
        #   counts are enabled and without otherwise. Each row is a sequence of strings.
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        # [FUNCTION DOC]
        # HELPER: abstract method returning an iterator of body rows including the non
        #   null count.
        """Iterator with string representation of body data with counts."""

    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        # [FUNCTION DOC]
        # HELPER: abstract method returning an iterator of body rows without the non
        #   null count.
        """Iterator with string representation of body data without counts."""

    def add_header_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends the header line: titles padded to the final column widths and
        #   joined with the spacing.
        header_line = self.SPACING.join(
            [
                _put_str(header, col_width)
                for header, col_width in zip(
                    self.headers, self.gross_column_widths, strict=True
                )
            ]
        )
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends the separator line: dashes as long as each title, padded to
        #   the final column widths and joined with the spacing.
        separator_line = self.SPACING.join(
            [
                _put_str("-" * header_colwidth, gross_colwidth)
                for header_colwidth, gross_colwidth in zip(
                    self.header_column_widths, self.gross_column_widths, strict=True
                )
            ]
        )
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends one line per body row, with cells padded to the final column
        #   widths and joined with the spacing.
        for row in self.strrows:
            body_line = self.SPACING.join(
                [
                    _put_str(col, gross_colwidth)
                    for col, gross_colwidth in zip(
                        row, self.gross_column_widths, strict=True
                    )
                ]
            )
            self._lines.append(body_line)

    def _gen_non_null_counts(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # HELPER: iterator over texts of the form count followed by non-null, one per
        #   column.
        """Iterator with string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f"{count} non-null"

    def _gen_dtypes(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # HELPER: iterator over the pretty printed dtype of each column.
        """Iterator with string representation of column dtypes."""
        for dtype in self.dtypes:
            yield pprint_thing(dtype)


class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    # [CLASS DOC]
    # HELPER: frame builder for the detailed per column table. State: info, with_counts
    #   (whether to include the non null count column), strrows (the body rows, computed
    #   at construction) and gross_column_widths (computed at construction). The report
    #   has the object type, index range, a total columns line, a header, a separator,
    #   body rows, dtype summary and optional memory usage.
    """
    Dataframe info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: DataFrameInfo,
        with_counts: bool,
    ) -> None:
        # [FUNCTION DOC]
        # Keyword only constructor. Inputs: info and with_counts. Stores them,
        #   materializes the body rows and computes the final column widths.
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends, in order, the object type, index range, columns summary,
        #   header, separator, body rows, dtypes and (if enabled) memory usage lines.
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    @property
    def headers(self) -> Sequence[str]:
        # [FUNCTION DOC]
        # HELPER: property returning the table titles: number, column, non null count
        #   (only with counts) and dtype.
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return [" # ", "Column", "Non-Null Count", "Dtype"]
        return [" # ", "Column", "Dtype"]

    def add_columns_summary_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends the line giving the total number of data columns.
        self._lines.append(f"Data columns (total {self.col_count} columns):")

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        # [FUNCTION DOC]
        # HELPER: iterator of rows pairing the column number, column label and dtype
        #   texts.
        """Iterator with string representation of body data without counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_dtypes(),
            strict=True,
        )

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        # [FUNCTION DOC]
        # HELPER: iterator of rows pairing the column number, column label, non null
        #   count text and dtype text.
        """Iterator with string representation of body data with counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_non_null_counts(),
            self._gen_dtypes(),
            strict=True,
        )

    def _gen_line_numbers(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # HELPER: iterator of column position texts, each with a leading space to align
        #   under the number title.
        """Iterator with string representation of column numbers."""
        for i, _ in enumerate(self.ids):
            yield f" {i}"

    def _gen_columns(self) -> Iterator[str]:
        # [FUNCTION DOC]
        # HELPER: iterator of the pretty printed column labels.
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)


class _SeriesTableBuilder(_TableBuilderAbstract):
    # [CLASS DOC]
    # HELPER: abstract builder for the series report. Holds the SeriesInfo, always
    #   produces the non empty report (a series has no empty case), and provides the
    #   series, and the memory usage line.
    """
    Abstract builder for series info table.

    Parameters
    ----------
    info : SeriesInfo.
        Instance of SeriesInfo.
    """

    def __init__(self, *, info: SeriesInfo) -> None:
        # [FUNCTION DOC]
        # Keyword only constructor taking the SeriesInfo, which is stored.
        self.info: SeriesInfo = info

    def get_lines(self) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: builds the report. Resets the list of lines, fills them with the non
        #   empty report and returns them.
        self._lines = []
        self._fill_non_empty_info()
        return self._lines

    @property
    def data(self) -> Series:
        # [FUNCTION DOC]
        # HELPER: property returning the series.
        """Series."""
        return self.info.data

    def add_memory_usage_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends a line with the memory usage text.
        """Add line containing memory usage."""
        self._lines.append(f"memory usage: {self.memory_usage_string}")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        # [FUNCTION DOC]
        # HELPER: abstract method that appends the report lines.
        """Add lines to the info table, pertaining to non-empty series."""


class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    # [CLASS DOC]
    # HELPER: series builder for compact output: object type, index range, dtype summary
    #   and optional memory usage.
    """
    Series info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends the object type, index range, dtypes and (if enabled) memory
        #   usage lines.
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()


class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    # [CLASS DOC]
    # HELPER: series builder for the detailed table. State: info, with_counts, strrows
    #   and gross_column_widths, computed at construction. The report has the object
    #   type, index range, series name, header, separator, body rows, dtype summary and
    #   optional memory usage.
    """
    Series info table builder for verbose output.
    """

    def __init__(
        self,
        *,
        info: SeriesInfo,
        with_counts: bool,
    ) -> None:
        # [FUNCTION DOC]
        # Keyword only constructor. Inputs: info and with_counts. Stores them,
        #   materializes the body rows and computes the final column widths.
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends the object type, index range, series name, header, separator,
        #   body, dtypes and (if enabled) memory usage lines.
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_series_name_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_series_name_line(self) -> None:
        # [FUNCTION DOC]
        # HELPER: appends a line with the series name.
        self._lines.append(f"Series name: {self.data.name}")

    @property
    def headers(self) -> Sequence[str]:
        # [FUNCTION DOC]
        # HELPER: property returning the titles: non null count (only with counts) and
        #   dtype.
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return ["Non-Null Count", "Dtype"]
        return ["Dtype"]

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        # [FUNCTION DOC]
        # HELPER: iterator of one element rows holding the dtype text.
        """Iterator with string representation of body data without counts."""
        yield from ([dtype] for dtype in self._gen_dtypes())

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        # [FUNCTION DOC]
        # HELPER: iterator of rows pairing the non null count text with the dtype text.
        """Iterator with string representation of body data with counts."""
        yield from zip(self._gen_non_null_counts(), self._gen_dtypes(), strict=True)


def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[str, int]:
    # [FUNCTION DOC]
    # HELPER: counts columns per dtype for a frame. Returns a mapping (a series) from
    #   dtype name to count, obtained by counting the dtypes and summing counts that
    #   share a name (so that, for example, all categorical dtypes with different
    #   categories are merged).
    """
    Create mapping between datatypes and their number of occurrences.
    """
    # groupby dtype.name to collect e.g. Categorical columns
    return df.dtypes.value_counts().groupby(lambda x: x.name).sum()
