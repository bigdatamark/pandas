# [MODULE DOC]
# Core of pandas Excel I/O: the public reader function, the reader base class, the
#   writer base class and the file wrapper class.
# Public contents: read_excel (function), ExcelFile (workbook wrapper that owns an
#   engine specific reader), ExcelWriter (abstract writer; instantiating it selects a
#   concrete engine subclass), and inspect_excel_format (sniffs a file format). Also
#   exported for internal use: BaseExcelReader (generic base class for the engine
#   readers in the sibling engine modules) and the signature constants used for
#   sniffing.
# Relationships: engine readers (calamine, odf, openpyxl, pyxlsb, xlrd) subclass
#   BaseExcelReader and are registered in the ExcelFile engine table; engine writers
#   subclass ExcelWriter and are looked up through the helper functions in the sibling
#   util module; parsing of the raw cell grid is delegated to the shared text parser
#   used by the CSV readers.
# Import-time side effects: none beyond importing the engine reader modules lazily
#   inside the ExcelFile class body, which happens when this module is imported (those
#   modules import their third party libraries only lazily).

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
import datetime
from decimal import Decimal
from functools import partial
import os
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings
import zipfile

from pandas._config import config

from pandas._libs import lib
from pandas.compat._optional import (
    get_version,
    import_optional_dependency,
)
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
    set_module,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_bool,
    is_decimal,
    is_file_like,
    is_float,
    is_integer,
    is_list_like,
)

from pandas.core.frame import DataFrame
from pandas.util.version import Version

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
from pandas.io.excel._util import (
    fill_mi_header,
    get_default_engine,
    get_writer,
    maybe_convert_usecols,
    pop_header_name,
)
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from types import TracebackType

    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        ExcelWriterIfSheetExists,
        FilePath,
        HashableT,
        IntStrT,
        ReadBuffer,
        SequenceNotStr,
        StorageOptions,
        WriteExcelBuffer,
    )


@overload
def read_excel(
    io,
    # sheet name is str or int -> DataFrame
    sheet_name: str | int = ...,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[HashableT], bool]
    | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame: ...
    # [FUNCTION DOC]
    # Typing overload only (no runtime behavior). Selected when the sheet selector is a
    #   single name or position; the static return type is a single table. See the
    #   implementation below.
    # KNOWN ISSUE: this overload (and the next) omits the engine_kwargs keyword that the
    #   implementation accepts, so type checkers reject calls passing it.


@overload
def read_excel(
    io,
    # sheet name is list or None -> dict[IntStrT, DataFrame]
    sheet_name: list[IntStrT] | None,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[HashableT], bool]
    | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> dict[IntStrT, DataFrame]: ...
    # [FUNCTION DOC]
    # Typing overload only (no runtime behavior). Selected when the sheet selector is a
    #   list or None; the static return type is a mapping from sheet selector to table.
    #   See the implementation below.
    # KNOWN ISSUE: like the first overload, it omits the engine_kwargs keyword that the
    #   implementation accepts.


@set_module("pandas")
def read_excel(
    io,
    sheet_name: str | int | list[IntStrT] | None = 0,
    *,
    header: int | Sequence[int] | None = 0,
    names: SequenceNotStr[Hashable] | range | None = None,
    index_col: int | str | Sequence[int] | None = None,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[HashableT], bool]
    | None = None,
    dtype: DtypeArg | None = None,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = None,
    converters: dict[str, Callable] | dict[int, Callable] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
    nrows: int | None = None,
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: list | dict | bool = False,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = ".",
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    engine_kwargs: dict | None = None,
) -> DataFrame | dict[IntStrT, DataFrame]:
    # [FUNCTION DOC]
    # Public entry point that reads one or more worksheets of an Excel or OpenDocument
    #   workbook into tables. Parameters mirror the CSV reader where applicable: io is a
    #   path, URL, file like object, ExcelFile or engine workbook; sheet_name selects
    #   sheets (name, zero based position, list of those, or None for all sheets;
    #   default first sheet); header, names, index_col, usecols, dtype, converters,
    #   true_values, false_values, skiprows, nrows, na_values, keep_default_na,
    #   na_filter, verbose, parse_dates, date_format, thousands, decimal, comment,
    #   skipfooter and dtype_backend are forwarded to the sheet parser; engine picks the
    #   reader (None means auto detect); storage_options is used when opening remote or
    #   special paths; engine_kwargs is passed to the engine when it loads the workbook
    #   (default none, treated as empty).
    # Returns a single table when sheet_name is a scalar, otherwise a mapping from each
    #   requested sheet selector to its table.
    # Behavior: first validates the dtype backend argument. If io is not already an
    #   ExcelFile, builds one (and remembers that it must close it afterwards); if it is
    #   an ExcelFile and an engine is given that differs from the file's engine, raises
    #   a value error because the engine is already fixed. Then delegates to the
    #   ExcelFile parse method and always closes the file in a finally step if this
    #   function opened it.
    """
    Read an Excel file into a ``DataFrame``.

    Supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions
    read from a local filesystem or URL. Supports an option to read
    a single sheet or a list of sheets.

    Parameters
    ----------
    io : str, ExcelFile, xlrd.Book, path object, or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: ``file://localhost/path/to/table.xlsx``.

        If you want to pass in a path object, pandas accepts any ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.

    sheet_name : str, int, list, or None, default 0
        Strings are used for sheet names. Integers are used in zero-indexed
        sheet positions (chart sheets do not count as a sheet position).
        Lists of strings/integers are used to request multiple sheets.
        When ``None``, will return a dictionary containing DataFrames for each sheet.

        Available cases:

        * Defaults to ``0``: 1st sheet as a `DataFrame`
        * ``1``: 2nd sheet as a `DataFrame`
        * ``"Sheet1"``: Load sheet with name "Sheet1"
        * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"
          as a dict of `DataFrame`
        * ``None``: Returns a dictionary containing DataFrames for each sheet.

    header : int, list of int, default 0
        Row (0-indexed) to use for the column labels of the parsed
        DataFrame. If a list of integers is passed those row positions will
        be combined into a ``MultiIndex``. Use None if there is no header.
    names : array-like, default None
        List of column names to use. If file contains no header row,
        then you should explicitly pass header=None.
    index_col : int, str, list of int, default None
        Column (0-indexed) to use as the row labels of the DataFrame.
        Pass None if there is no such column.  If a list is passed,
        those columns will be combined into a ``MultiIndex``.  If a
        subset of data is selected with ``usecols``, index_col
        is based on the subset.

        Missing values will be forward filled to allow roundtripping with
        ``to_excel`` for ``merged_cells=True``. To avoid forward filling the
        missing values use ``set_index`` after reading the data instead of
        ``index_col``.
    usecols : str, list-like, or callable, default None
        * If None, then parse all columns.
        * If str, then indicates comma separated list of Excel column letters
          and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of
          both sides.
        * If list of int, then indicates list of column numbers to be parsed
          (0-indexed).
        * If list of string, then indicates list of column names to be parsed.
        * If callable, then evaluate each column name against it and parse the
          column if the callable returns ``True``.

        Returns a subset of the columns according to behavior above.
    dtype : Type name or dict of column -> type, default None
        Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32}
        Use ``object`` to preserve data as stored in Excel and not interpret dtype,
        which will necessarily result in ``object`` dtype.
        If converters are specified, they will be applied INSTEAD
        of dtype conversion.
        If you use ``None``, it will infer the dtype of each column based on the data.
    engine : {'openpyxl', 'calamine', 'odf', 'pyxlsb', 'xlrd'}, default None
        If io is not a buffer or path, this must be set to identify io.
        Engine compatibility :

        - ``openpyxl`` supports newer Excel file formats.
        - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
          and OpenDocument (.ods) file formats.
        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
        - ``pyxlsb`` supports Binary Excel files.
        - ``xlrd`` supports old-style Excel files (.xls).

        When ``engine=None``, the following logic will be used to determine the engine:

        - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
          then `odf <https://pypi.org/project/odfpy/>`_ will be used.
        - Otherwise if ``path_or_buffer`` is an xls format, ``xlrd`` will be used.
        - Otherwise if ``path_or_buffer`` is in xlsb format, ``pyxlsb`` will be used.
        - Otherwise ``openpyxl`` will be used.

    converters : dict, default None
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the Excel cell content, and return the transformed
        content.
    true_values : list, default None
        Values to consider as True.
    false_values : list, default None
        Values to consider as False.
    skiprows : list-like, int, or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int) at the
        start of the file. If callable, the callable function will be evaluated
        against the row indices, returning True if the row should be skipped and
        False otherwise. An example of a valid callable argument would be ``lambda
        x: x in [0, 2]``.
    nrows : int, default None
        Number of rows to parse. Does not include header rows.
    na_values : scalar, str, list-like, or dict, default None
        Additional strings to recognize as NA/NaN. If dict passed, specific
        per-column NA values. By default the following values are interpreted
        as NaN: '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
        '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None',
        'n/a', 'nan', 'null'.
    keep_default_na : bool, default True
        Whether or not to include the default NaN values when parsing the data.
        Depending on whether ``na_values`` is passed in, the behavior is as follows:

        * If ``keep_default_na`` is True, and ``na_values`` are specified,
          ``na_values`` is appended to the default NaN values used for parsing.
        * If ``keep_default_na`` is True, and ``na_values`` are not specified, only
          the default NaN values are used for parsing.
        * If ``keep_default_na`` is False, and ``na_values`` are specified, only
          the NaN values specified ``na_values`` are used for parsing.
        * If ``keep_default_na`` is False, and ``na_values`` are not specified, no
          strings will be parsed as NaN.

        Note that if `na_filter` is passed in as False, the ``keep_default_na`` and
        ``na_values`` parameters will be ignored.
    na_filter : bool, default True
        Detect missing value markers (empty strings and the value of na_values). In
        data without any NAs, passing ``na_filter=False`` can improve the
        performance of reading a large file.
    verbose : bool, default False
        Indicate number of NA values placed in non-numeric columns.
    parse_dates : bool, list-like, or dict, default False
        The behavior is as follows:

        * ``bool``. If True -> try parsing the index.
        * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
          each as a separate date column.
        * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
          a single date column.
        * ``dict``, e.g. {'foo' : [1, 3]} -> parse columns 1, 3 as date and call
          result 'foo'

        If a column or index contains an unparsable date, the entire column or
        index will be returned unaltered as an object data type. If you don`t want to
        parse some cells as date just change their type in Excel to "Text".
        For non-standard datetime parsing, use ``pd.to_datetime`` after
        ``pd.read_excel``.

        Note: A fast-path exists for iso8601-formatted dates.
    date_format : str or dict of column -> format, default ``None``
        If used in conjunction with ``parse_dates``, will parse dates according to this
        format. For anything more complex,
        please read in as ``object`` and then apply :func:`to_datetime` as-needed.

        .. versionadded:: 2.0.0

    thousands : str, default None
        Thousands separator for parsing string columns to numeric.  Note that
        this parameter is only necessary for columns stored as TEXT in Excel,
        any numeric columns will automatically be parsed, regardless of display
        format.
    decimal : str, default '.'
        Character to recognize as decimal point for parsing string columns to numeric.
        Note that this parameter is only necessary for columns stored as TEXT in Excel,
        any numeric columns will automatically be parsed, regardless of display
        format.(e.g. use ',' for European data).
    comment : str, default None
        Comments out remainder of line. Pass a character or characters to this
        argument to indicate comments in the input file. Any data between the
        comment string and the end of the current line is ignored.
    skipfooter : int, default 0
        Rows at the end to skip (0-indexed).
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable

        :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    engine_kwargs : dict, optional
        Arbitrary keyword arguments passed to excel engine.

    Returns
    -------
    DataFrame or dict of DataFrames
        DataFrame from the passed in Excel file. See notes in sheet_name
        argument for more information on when a dict of DataFrames is returned.

    See Also
    --------
    DataFrame.to_excel : Write DataFrame to an Excel file.
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Notes
    -----
    For specific information on the methods used for each Excel engine, refer to the
    pandas
    :ref:`user guide <io.excel_reader>`

    Examples
    --------
    The file can be read using the file name as string or an open file object:

    >>> pd.read_excel("tmp.xlsx", index_col=0)  # doctest: +SKIP
           Name  Value
    0   string1      1
    1   string2      2
    2  #Comment      3

    >>> pd.read_excel(open("tmp.xlsx", "rb"), sheet_name="Sheet3")  # doctest: +SKIP
       Unnamed: 0      Name  Value
    0           0   string1      1
    1           1   string2      2
    2           2  #Comment      3

    Index and header can be specified via the `index_col` and `header` arguments

    >>> pd.read_excel("tmp.xlsx", index_col=None, header=None)  # doctest: +SKIP
         0         1      2
    0  NaN      Name  Value
    1  0.0   string1      1
    2  1.0   string2      2
    3  2.0  #Comment      3

    Column types are inferred but can be explicitly specified

    >>> pd.read_excel(
    ...     "tmp.xlsx", index_col=0, dtype={"Name": str, "Value": float}
    ... )  # doctest: +SKIP
           Name  Value
    0   string1    1.0
    1   string2    2.0
    2  #Comment    3.0

    True, False, and NA values, and thousands separators have defaults,
    but can be explicitly specified, too. Supply the values you would like
    as strings or lists of strings!

    >>> pd.read_excel(
    ...     "tmp.xlsx", index_col=0, na_values=["string1", "string2"]
    ... )  # doctest: +SKIP
           Name  Value
    0       NaN      1
    1       NaN      2
    2  #Comment      3

    Comment lines in the excel input file can be skipped using the
    ``comment`` kwarg.

    >>> pd.read_excel("tmp.xlsx", index_col=0, comment="#")  # doctest: +SKIP
          Name  Value
    0  string1    1.0
    1  string2    2.0
    2     None    NaN
    """
    check_dtype_backend(dtype_backend)
    should_close = False
    if engine_kwargs is None:
        engine_kwargs = {}

    if not isinstance(io, ExcelFile):
        should_close = True
        io = ExcelFile(
            io,
            storage_options=storage_options,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    elif engine and engine != io.engine:
        raise ValueError(
            "Engine should not be specified when passing "
            "an ExcelFile - ExcelFile already has the engine set"
        )

    try:
        data = io.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            verbose=verbose,
            parse_dates=parse_dates,
            date_format=date_format,
            thousands=thousands,
            decimal=decimal,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
        )
    finally:
        # make sure to close opened file handles
        if should_close:
            io.close()
    return data


_WorkbookT = TypeVar("_WorkbookT")


class BaseExcelReader(Generic[_WorkbookT]):
    # [CLASS DOC]
    # Generic (over the engine workbook type) abstract base class for engine specific
    #   Excel readers. Subclasses must supply the workbook class, a method that loads a
    #   workbook from a binary handle, the list of sheet names, per sheet lookup by name
    #   and by position, and a method that materializes a sheet into a list of rows.
    # State: book is the loaded engine workbook object; handles is the IOHandles object
    #   wrapping the opened file (or the already open object) and is closed by the close
    #   method.
    # Key methods: the constructor (opens/loads), close, parse (public, multi sheet
    #   driver), and the private helpers that translate skiprows/nrows into a row count
    #   to read and that turn the raw cell grid into a table.
    book: _WorkbookT

    def __init__(
        self,
        filepath_or_buffer,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Open the workbook. Parameters: filepath_or_buffer is a path, URL, binary
        #   buffer, ExcelFile, or an engine workbook object; storage_options is
        #   forwarded to the file opener; engine_kwargs (default none, treated as empty)
        #   is forwarded to the workbook loader.
        # Behavior: wraps the input in a handles object with no compression. Unless the
        #   input is an ExcelFile or already an instance of the engine workbook class,
        #   opens it in binary mode through the common file handle opener. If the
        #   resulting handle is itself an engine workbook it is used directly as the
        #   book; otherwise if it has a read method the stream is rewound and the book
        #   is loaded through the load_workbook hook (on any failure the reader is
        #   closed and the exception re raised). Anything else raises a value error
        #   saying the engine must be set explicitly when not passing a buffer or path.
        # Side effects: may open a file handle that remains open until close is called.
        if engine_kwargs is None:
            engine_kwargs = {}

        self.handles = IOHandles(
            handle=filepath_or_buffer, compression={"method": None}
        )
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(
                filepath_or_buffer, "rb", storage_options=storage_options, is_text=False
            )

        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
        elif hasattr(self.handles.handle, "read"):
            # N.B. xlrd.Book has a read attribute too
            self.handles.handle.seek(0)
            try:
                self.book = self.load_workbook(self.handles.handle, engine_kwargs)
            except Exception:
                self.close()
                raise
        else:
            raise ValueError(
                "Must explicitly set engine if not passing in buffer or path for io."
            )

    @property
    def _workbook_class(self) -> type[_WorkbookT]:
        # [FUNCTION DOC]
        # Abstract property: returns the engine's workbook class, used to recognize pre-
        #   loaded workbook objects. Raises not implemented error in the base class.
        raise NotImplementedError

    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> _WorkbookT:
        # [FUNCTION DOC]
        # Abstract hook: load and return an engine workbook from a binary file like
        #   object, given the engine_kwargs mapping. Raises not implemented error in the
        #   base class.
        raise NotImplementedError

    def close(self) -> None:
        # [FUNCTION DOC]
        # Release resources. If a book was loaded, closes it when it has a close method
        #   (needed by the pyxlsb and openpyxl engines which hold temporary files or
        #   file handles), otherwise calls the release resources method that the xlrd
        #   workbook offers. Then closes the underlying handles. Safe to call when
        #   construction failed before the book was assigned. Returns nothing.
        if hasattr(self, "book"):
            if hasattr(self.book, "close"):
                # pyxlsb: opens a TemporaryFile
                # openpyxl: https://stackoverflow.com/questions/31416842/
                #     openpyxl-does-not-close-excel-workbook-in-read-only-mode
                self.book.close()
            elif hasattr(self.book, "release_resources"):
                # xlrd
                # https://github.com/python-excel/xlrd/blob/2.0.1/xlrd/book.py#L548
                self.book.release_resources()
        self.handles.close()

    @property
    def sheet_names(self) -> list[str]:
        # [FUNCTION DOC]
        # Abstract property: returns the list of sheet names of the workbook. Raises not
        #   implemented error in the base class.
        raise NotImplementedError

    def get_sheet_by_name(self, name: str):
        # [FUNCTION DOC]
        # Abstract hook: returns the engine sheet object with the given name. Raises not
        #   implemented error in the base class.
        raise NotImplementedError

    def get_sheet_by_index(self, index: int):
        # [FUNCTION DOC]
        # Abstract hook: returns the engine sheet object at the given zero based
        #   position. Raises not implemented error in the base class.
        raise NotImplementedError

    def get_sheet_data(self, sheet, rows: int | None = None):
        # [FUNCTION DOC]
        # Abstract hook: converts a sheet into a list of rows (each a list of Python
        #   cell values). The optional rows argument (default none) limits how many rows
        #   are read; none reads all. Raises not implemented error in the base class.
        raise NotImplementedError

    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        # [FUNCTION DOC]
        # Validation helper for subclasses. Input: a zero based sheet position. Raises a
        #   value error naming the position and the number of worksheets found when the
        #   position is not smaller than the number of sheets. Negative positions are
        #   not rejected here. Returns nothing.
        n_sheets = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(
                f"Worksheet index {index} is invalid, {n_sheets} worksheets found"
            )

    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        # [FUNCTION DOC]
        # Validation helper for subclasses. Input: a sheet name. Raises a value error
        #   saying the worksheet was not found when the name is not in the list of sheet
        #   names. Returns nothing.
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def _check_skiprows_func(
        self,
        skiprows: Callable,
        rows_to_use: int,
    ) -> int:
        # [FUNCTION DOC]
        # HELPER: used by the row count calculation when skiprows is callable.
        # Determines how many file rows must be read so that the requested number of
        #   rows (header plus data, given by rows_to_use) remain after rows for which
        #   the callable returns true are dropped. Walks row indices from zero, counting
        #   rows the callable does not skip, and returns the row index count reached
        #   when enough rows were counted. Does not terminate if the callable skips
        #   every row indefinitely.
        """
        Determine how many file rows are required to obtain `nrows` data
        rows when `skiprows` is a function.

        Parameters
        ----------
        skiprows : function
            The function passed to read_excel by the user.
        rows_to_use : int
            The number of rows that will be needed for the header and
            the data.

        Returns
        -------
        int
        """
        i = 0
        rows_used_so_far = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def _calc_rows(
        self,
        header: int | Sequence[int] | None,
        index_col: int | Sequence[int] | None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None,
        nrows: int | None,
    ) -> int | None:
        # [FUNCTION DOC]
        # HELPER: used by the parse method to avoid reading more of the sheet than
        #   needed.
        # Inputs: header (row position, list of positions, or none), index_col, skiprows
        #   (none, count, list of positions or callable) and nrows. Returns none when
        #   nrows is none (read everything). Otherwise computes the number of header
        #   rows (one if no header, otherwise one plus the last header position), adds
        #   one more when a multi row header is combined with an index column (the row
        #   holding index names), then: with no skiprows returns header rows plus nrows;
        #   with an integer skiprows adds that count; with a list or callable delegates
        #   to the callable skip counter so skipped rows are not counted; for any other
        #   skiprows type returns none so no optimization is applied.
        """
        If nrows specified, find the number of rows needed from the
        file, otherwise return None.


        Parameters
        ----------
        header : int, list of int, or None
            See read_excel docstring.
        index_col : int, str, list of int, or None
            See read_excel docstring.
        skiprows : list-like, int, callable, or None
            See read_excel docstring.
        nrows : int or None
            See read_excel docstring.

        Returns
        -------
        int or None
        """
        if nrows is None:
            return None
        if header is None:
            header_rows = 1
        elif is_integer(header):
            header = cast(int, header)
            header_rows = 1 + header
        else:
            header = cast(Sequence, header)
            header_rows = 1 + header[-1]
        # If there is a MultiIndex header and an index then there is also
        # a row containing just the index name(s)
        if is_list_like(header) and index_col is not None:
            header = cast(Sequence, header)
            if len(header) > 1:
                header_rows += 1
        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):

            def f(skiprows: Sequence, x: int) -> bool:
                # [FUNCTION DOC]
                # HELPER: nested predicate used with a partial application to turn a
                #   list of skipped positions into a callable. Returns true if the row
                #   index x is contained in the sequence skiprows.
                return x in skiprows

            skiprows = cast(Sequence, skiprows)
            return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(
                skiprows,
                header_rows + nrows,
            )
        # else unexpected skiprows type: read_excel will not optimize
        # the number of rows read from file
        return None

    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        dtype: DtypeArg | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values=None,
        verbose: bool = False,
        parse_dates: list | dict | bool = False,
        date_format: dict[Hashable, str] | str | None = None,
        thousands: str | None = None,
        decimal: str = ".",
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ):
        # [FUNCTION DOC]
        # Public driver that reads one or several sheets and returns tables. Parameters
        #   follow read_excel: sheet_name (name, position, list, or none for all sheets;
        #   default first sheet), header, names, index_col, usecols, dtype, true_values,
        #   false_values, skiprows, nrows, na_values, verbose (print a line per sheet),
        #   parse_dates, date_format, thousands, decimal, comment, skipfooter,
        #   dtype_backend, and extra keyword arguments passed through to the text
        #   parser.
        # Returns a mapping of sheet selector to table when sheet_name is a list or
        #   none, otherwise the single table for the last (only) selected sheet.
        # Behavior: validates the header argument and that nrows is a valid integer;
        #   builds the list of selected sheets (none means all sheet names; duplicates
        #   of the same value are removed while keeping order); for each sheet looks it
        #   up by name or by position, computes how many rows are needed, reads the
        #   sheet data, closes the sheet if it supports closing, normalizes usecols once
        #   through the util helper, stores an empty table if the sheet has no data, and
        #   otherwise delegates to the private sheet parser. Raises a value error if the
        #   list of sheets is empty.
        # Side effects: prints a progress line per sheet when verbose is true.
        validate_header_arg(header)
        validate_integer("nrows", nrows)

        ret_dict = False

        # Keep sheetname to maintain backwards compatibility.
        sheets: list[int] | list[str]
        if isinstance(sheet_name, list):
            sheets = sheet_name
            ret_dict = True
        elif sheet_name is None:
            sheets = self.sheet_names
            ret_dict = True
        elif isinstance(sheet_name, str):
            sheets = [sheet_name]
        else:
            sheets = [sheet_name]

        # handle same-type duplicates.
        sheets = cast(Union[list[int], list[str]], list(dict.fromkeys(sheets).keys()))

        output = {}

        last_sheetname = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f"Reading sheet {asheetname}")

            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:  # assume an integer if not a string
                sheet = self.get_sheet_by_index(asheetname)

            file_rows_needed = self._calc_rows(header, index_col, skiprows, nrows)
            data = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, "close"):
                # pyxlsb opens two TemporaryFiles
                sheet.close()
            usecols = maybe_convert_usecols(usecols)

            if not data:
                output[asheetname] = DataFrame()
                continue

            output = self._parse_sheet(
                data=data,
                output=output,
                asheetname=asheetname,
                header=header,
                names=names,
                index_col=index_col,
                usecols=usecols,
                dtype=dtype,
                skiprows=skiprows,
                nrows=nrows,
                true_values=true_values,
                false_values=false_values,
                na_values=na_values,
                parse_dates=parse_dates,
                date_format=date_format,
                thousands=thousands,
                decimal=decimal,
                comment=comment,
                skipfooter=skipfooter,
                dtype_backend=dtype_backend,
                **kwds,
            )

        if last_sheetname is None:
            raise ValueError("Sheet name is an empty list")

        if ret_dict:
            return output
        else:
            return output[last_sheetname]

    def _parse_sheet(
        self,
        data: list,
        output: dict,
        asheetname: str | int | None = None,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        dtype: DtypeArg | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        na_values=None,
        parse_dates: list | dict | bool = False,
        date_format: dict[Hashable, str] | str | None = None,
        thousands: str | None = None,
        decimal: str = ".",
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ):
        # [FUNCTION DOC]
        # HELPER: turns one sheet's raw rows into a table and stores it in the shared
        #   output mapping (under key asheetname), returning that mapping.
        # Inputs: data is the list of rows (mutated in place for header and index
        #   forward filling); output is the accumulating mapping; asheetname is the
        #   sheet key; remaining arguments are the read options forwarded to the shared
        #   text parser, plus extra keyword arguments.
        # Behavior: normalizes a length one list header to a scalar; for a multi row
        #   header forward fills merged header cells row by row (raising a value error
        #   when a header position, after adding an integer skiprows offset, is beyond
        #   the data) and, when an index column is set, pops the header name for the
        #   index; detects whether a row of index names follows a multi level header
        #   (all non-index, non-filled cells empty); forward fills blanks in the index
        #   columns of a multi index; then builds a text parser with blank line skipping
        #   off and reads nrows rows. If header names were collected, they are set as
        #   the column level names. An empty data error produces an empty table; any
        #   other exception has the sheet name appended to its first argument and is re
        #   raised.
        is_list_header = False
        is_len_one_list_header = False
        if is_list_like(header):
            assert isinstance(header, Sequence)
            is_list_header = True
            if len(header) == 1:
                is_len_one_list_header = True

        if is_len_one_list_header:
            header = cast(Sequence[int], header)[0]

        # forward fill and pull out names for MultiIndex column
        header_names = None
        if header is not None and is_list_like(header):
            assert isinstance(header, Sequence)

            header_names = []
            control_row = [True] * len(data[0])

            for row in header:
                if is_integer(skiprows):
                    assert isinstance(skiprows, int)
                    row += skiprows

                if row > len(data) - 1:
                    raise ValueError(
                        f"header index {row} exceeds maximum index "
                        f"{len(data) - 1} of data.",
                    )

                data[row], control_row = fill_mi_header(data[row], control_row)

                if index_col is not None:
                    header_name, _ = pop_header_name(data[row], index_col)
                    header_names.append(header_name)

        # If there is a MultiIndex header and an index then there is also
        # a row containing just the index name(s)
        has_index_names = False
        if is_list_header and not is_len_one_list_header and index_col is not None:
            index_col_set: set[int]
            if isinstance(index_col, int):
                index_col_set = {index_col}
            else:
                assert isinstance(index_col, Sequence)
                index_col_set = set(index_col)

            # We have to handle mi without names. If any of the entries in the data
            # columns are not empty, this is a regular row
            assert isinstance(header, Sequence)
            if len(header) < len(data):
                potential_index_names = data[len(header)]
                has_index_names = all(
                    x == "" or x is None
                    for i, x in enumerate(potential_index_names)
                    if not control_row[i] and i not in index_col_set
                )

        if is_list_like(index_col):
            # Forward fill values for MultiIndex index.
            if header is None:
                offset = 0
            elif isinstance(header, int):
                offset = 1 + header
            else:
                offset = 1 + max(header)

            # GH34673: if MultiIndex names present and not defined in the header,
            # offset needs to be incremented so that forward filling starts
            # from the first MI value instead of the name
            if has_index_names:
                offset += 1

            # Check if we have an empty dataset
            # before trying to collect data.
            if offset < len(data):
                assert isinstance(index_col, Sequence)

                for col in index_col:
                    last = data[offset][col]

                    for row in range(offset + 1, len(data)):
                        if data[row][col] == "" or data[row][col] is None:
                            data[row][col] = last
                        else:
                            last = data[row][col]

        # GH 12292 : error when read one empty column from excel file
        try:
            parser = TextParser(
                data,
                names=names,
                header=header,
                index_col=index_col,
                has_index_names=has_index_names,
                dtype=dtype,
                true_values=true_values,
                false_values=false_values,
                skiprows=skiprows,
                nrows=nrows,
                na_values=na_values,
                skip_blank_lines=False,  # GH 39808
                parse_dates=parse_dates,
                date_format=date_format,
                thousands=thousands,
                decimal=decimal,
                comment=comment,
                skipfooter=skipfooter,
                usecols=usecols,
                dtype_backend=dtype_backend,
                **kwds,
            )

            output[asheetname] = parser.read(nrows=nrows)

            if header_names:
                output[asheetname].columns = output[asheetname].columns.set_names(
                    header_names
                )

        except EmptyDataError:
            # No Data, return an empty DataFrame
            output[asheetname] = DataFrame()

        except Exception as err:
            err.args = (f"{err.args[0]} (sheet: {asheetname})", *err.args[1:])
            raise err

        return output


@set_module("pandas")
class ExcelWriter(Generic[_WorkbookT]):
    # [CLASS DOC]
    # Abstract base class and factory for writing tables to Excel sheets. Calling the
    #   class directly (not a subclass) picks a concrete engine subclass from the file
    #   extension and the engine configuration options (or the explicit engine argument)
    #   and returns an instance of that subclass. Concrete engine subclasses must define
    #   the engine name, the supported extensions, the cells writing method, the save
    #   method, and expose the workbook and sheets.
    # State: _handles (IOHandles for the output), _cur_sheet, _date_format and
    #   _datetime_format (format strings, defaults for dates and datetimes), _mode
    #   (binary file mode used to open the output, with append mapped to read/update)
    #   and _if_sheet_exists (policy applied when a sheet exists in append mode).
    # Usable as a context manager; leaving the block saves and closes the output.
    """
    Class for writing DataFrame objects into excel sheets.

    Default is to use:

    * `xlsxwriter <https://pypi.org/project/XlsxWriter/>`__ for xlsx files if xlsxwriter
      is installed otherwise `openpyxl <https://pypi.org/project/openpyxl/>`__
    * `odf <https://pypi.org/project/odfpy/>`__ for ods files

    See :meth:`DataFrame.to_excel` for typical usage.

    The writer should be used as a context manager. Otherwise, call `close()` to save
    and close any opened file handles.

    Parameters
    ----------
    path : str or typing.BinaryIO
        Path to xls or xlsx or ods file.
    engine : str (optional)
        Engine to use for writing. If None, defaults to
        ``io.excel.<extension>.writer``.  NOTE: can only be passed as a keyword
        argument.
    date_format : str, default None
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
    datetime_format : str, default None
        Format string for datetime objects written into Excel files.
        (e.g. 'YYYY-MM-DD HH:MM:SS').
    mode : {'w', 'a'}, default 'w'
        File mode to use (write or append). Append does not work with fsspec URLs.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

    if_sheet_exists : {'error', 'new', 'replace', 'overlay'}, default 'error'
        How to behave when trying to write to a sheet that already
        exists (append mode only).

        * error: raise a ValueError.
        * new: Create a new sheet, with a name determined by the engine.
        * replace: Delete the contents of the sheet before writing to it.
        * overlay: Write contents to the existing sheet without first removing,
          but possibly over top of, the existing contents.

    engine_kwargs : dict, optional
        Keyword arguments to be passed into the engine. These will be passed to
        the following functions of the respective engines:

        * xlsxwriter: ``xlsxwriter.Workbook(file, **engine_kwargs)``
        * openpyxl (write mode): ``openpyxl.Workbook(**engine_kwargs)``
        * openpyxl (append mode): ``openpyxl.load_workbook(file, **engine_kwargs)``
        * odf: ``odf.opendocument.OpenDocumentSpreadsheet(**engine_kwargs)``

    See Also
    --------
    read_excel : Read an Excel sheet values (xlsx) file into DataFrame.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Notes
    -----
    For compatibility with CSV writers, ExcelWriter serializes lists
    and dicts to strings before writing.

    Examples
    --------
    Default usage:

    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    To write to separate sheets in a single file:

    >>> df1 = pd.DataFrame([["AAA", "BBB"]], columns=["Spam", "Egg"])  # doctest: +SKIP
    >>> df2 = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP
    ...     df2.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP

    You can set the date format or datetime format:

    >>> from datetime import date, datetime  # doctest: +SKIP
    >>> df = pd.DataFrame(
    ...     [
    ...         [date(2014, 1, 31), date(1999, 9, 24)],
    ...         [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
    ...     ],
    ...     index=["Date", "Datetime"],
    ...     columns=["X", "Y"],
    ... )  # doctest: +SKIP
    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     date_format="YYYY-MM-DD",
    ...     datetime_format="YYYY-MM-DD HH:MM:SS",
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    You can also append to an existing Excel file:

    >>> with pd.ExcelWriter("path_to_file.xlsx", mode="a", engine="openpyxl") as writer:
    ...     df.to_excel(writer, sheet_name="Sheet3")  # doctest: +SKIP

    Here, the `if_sheet_exists` parameter can be set to replace a sheet if it
    already exists:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="replace",
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP

    You can also write multiple DataFrames to a single sheet. Note that the
    ``if_sheet_exists`` parameter needs to be set to ``overlay``:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="overlay",
    ... ) as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")
    ...     df2.to_excel(writer, sheet_name="Sheet1", startcol=3)  # doctest: +SKIP

    You can store Excel file in RAM:

    >>> import io
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> buffer = io.BytesIO()
    >>> with pd.ExcelWriter(buffer) as writer:
    ...     df.to_excel(writer)

    You can pack Excel file into zip archive:

    >>> import zipfile  # doctest: +SKIP
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with zipfile.ZipFile("path_to_file.zip", "w") as zf:
    ...     with zf.open("filename.xlsx", "w") as buffer:
    ...         with pd.ExcelWriter(buffer) as writer:
    ...             df.to_excel(writer)  # doctest: +SKIP

    You can specify additional arguments to the underlying engine:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="xlsxwriter",
    ...     engine_kwargs={"options": {"nan_inf_to_errors": True}},
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    In append mode, ``engine_kwargs`` are passed through to
    openpyxl's ``load_workbook``:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="openpyxl",
    ...     mode="a",
    ...     engine_kwargs={"keep_vba": True},
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP
    """

    # Defining an ExcelWriter implementation (see abstract methods for more...)

    # - Mandatory
    #   - ``write_cells(self, cells, sheet_name=None, startrow=0, startcol=0)``
    #     --> called to write additional DataFrames to disk
    #   - ``_supported_extensions`` (tuple of supported extensions), used to
    #      check that engine supports the given extension.
    #   - ``_engine`` - string that gives the engine name. Necessary to
    #     instantiate class directly and bypass ``ExcelWriterMeta`` engine
    #     lookup.
    #   - ``save(self)`` --> called to save file to disk
    # - Mostly mandatory (i.e. should at least exist)
    #   - book, cur_sheet, path

    # - Optional:
    #   - ``__init__(self, path, engine=None, **kwargs)`` --> always called
    #     with path as first argument.

    # You also need to register the class with ``register_writer()``.
    # Technically, ExcelWriter implementations don't need to subclass
    # ExcelWriter.

    _engine: str
    _supported_extensions: tuple[str, ...]

    def __new__(
        cls,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict | None = None,
    ) -> Self:
        # [FUNCTION DOC]
        # Factory hook that chooses the concrete class. Inputs mirror the constructor.
        #   When called on the base class itself: if engine is none or the string auto,
        #   derives the extension from a string path (otherwise assumes xlsx), looks up
        #   the configured writer engine for that extension and, if that is auto,
        #   resolves a default engine; an unknown extension (no such configuration
        #   option) raises a value error saying there is no engine for the file type.
        #   Then resolves the class through the util registry lookup. When called on a
        #   subclass the class is used unchanged. Returns a new uninitialized instance
        #   of the resulting class (the constructor then runs).
        # only switch class if generic(ExcelWriter)
        if cls is ExcelWriter:
            if engine is None or (isinstance(engine, str) and engine == "auto"):
                if isinstance(path, str):
                    ext = os.path.splitext(path)[-1][1:]
                else:
                    ext = "xlsx"

                try:
                    engine = config.get_option(f"io.excel.{ext}.writer")
                    if engine == "auto":
                        engine = get_default_engine(ext, mode="writer")
                except KeyError as err:
                    raise ValueError(f"No engine for filetype: '{ext}'") from err

            # for mypy
            assert engine is not None
            #  error: Incompatible types in assignment (expression has type
            #  "type[ExcelWriter[Any]]", variable has type "type[Self]")
            cls = get_writer(engine)  # type: ignore[assignment]

        return object.__new__(cls)

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Read only property: the tuple of file extensions the writer engine supports.
        """Extensions that writer engine supports."""
        return self._supported_extensions

    @property
    def engine(self) -> str:
        # [FUNCTION DOC]
        # Read only property: the engine name string of this writer.
        """Name of engine."""
        return self._engine

    @property
    def sheets(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract property: mapping from sheet names to engine sheet objects. Raises
        #   not implemented error in the base class.
        """Mapping of sheet names to sheet objects."""
        raise NotImplementedError

    @property
    def book(self) -> _WorkbookT:
        # [FUNCTION DOC]
        # Abstract property: the engine workbook object, allowing access to engine
        #   specific features. Raises not implemented error in the base class.
        """
        Book instance. Class type will depend on the engine used.

        This attribute can be used to access engine-specific features.
        """
        raise NotImplementedError

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
        autofilter_range: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: engine hook called by the frame to Excel formatter to write cells.
        # Abstract: subclasses write the formatted cells (an iterable of cell objects)
        #   into the sheet named sheet_name (default none means the current sheet), with
        #   the upper left corner at startrow and startcol (zero based, default zero),
        #   optionally freezing panes (pair of row and column counts) and applying an
        #   automatic filter over a cell range string. Raises not implemented error in
        #   the base class.
        # KNOWN ISSUE: the docstring describes freeze_panes as the bottom most row and
        #   right most column to freeze; engines treat it as the number of rows and
        #   columns to freeze from the top left.
        """
        Write given formatted cells into Excel an excel sheet

        Parameters
        ----------
        cells : generator
            cell of formatted data to save to Excel sheet
        sheet_name : str, default None
            Name of Excel sheet, if None, then use self.cur_sheet
        startrow : upper left cell row to dump data frame
        startcol : upper left cell column to dump data frame
        freeze_panes: int tuple of length 2
            contains the bottom-most row and right-most column to freeze
        autofilter_range: str, default None
            column ranges to add automatic filters to, for example "A1:D5"
        """
        raise NotImplementedError

    def _save(self) -> None:
        # [FUNCTION DOC]
        # HELPER: engine hook that writes the workbook to the output handle. Abstract;
        #   raises not implemented error in the base class. Called by close.
        """
        Save workbook to disk.
        """
        raise NotImplementedError

    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Initialize the writer. Inputs: path (string path, binary buffer, or another
        #   ExcelWriter), engine (unused here, consumed by the factory), date_format and
        #   datetime_format (format strings; defaults year-month-day and year-month-day
        #   hour:minute:second), mode (write or append, default write), storage_options,
        #   if_sheet_exists (one of error, new, replace, overlay or none; none becomes
        #   error), engine_kwargs (unused here, consumed by subclasses).
        # Behavior: for a string path checks that the extension is supported, raising a
        #   value error otherwise. Forces a binary mode and maps append to read plus
        #   update so the engine can first read the existing file. Raises a value error
        #   if if_sheet_exists is not one of the valid values, or if it is given while
        #   the mode is not append. Wraps the path in a handles object, and unless the
        #   input is an ExcelWriter opens the file through the common opener in binary
        #   mode.
        # Side effects: opens a file handle that is released by close.
        # KNOWN ISSUE: the placeholder IOHandles built for a non-file input is given a
        #   compression dict keyed by the word compression, while the reader class uses
        #   a key named method; the two spellings are inconsistent and only the latter
        #   matches the compression options structure.
        # validate that this engine can handle the extension
        if isinstance(path, str):
            ext = os.path.splitext(path)[-1]
            self.check_extension(ext)

        # use mode to open the file
        if "b" not in mode:
            mode += "b"
        # use "a" for the user to append data to excel but internally use "r+" to let
        # the excel backend first read the existing file and then write any data to it
        mode = mode.replace("a", "r+")

        if if_sheet_exists not in (None, "error", "new", "replace", "overlay"):
            raise ValueError(
                f"'{if_sheet_exists}' is not valid for if_sheet_exists. "
                "Valid options are 'error', 'new', 'replace' and 'overlay'."
            )
        if if_sheet_exists and "r+" not in mode:
            raise ValueError("if_sheet_exists is only valid in append mode (mode='a')")
        if if_sheet_exists is None:
            if_sheet_exists = "error"
        self._if_sheet_exists = if_sheet_exists

        # cast ExcelWriter to avoid adding 'if self._handles is not None'
        self._handles = IOHandles(
            cast(IO[bytes], path), compression={"compression": None}
        )
        if not isinstance(path, ExcelWriter):
            self._handles = get_handle(
                path, mode, storage_options=storage_options, is_text=False
            )
        self._cur_sheet = None

        if date_format is None:
            self._date_format = "YYYY-MM-DD"
        else:
            self._date_format = date_format
        if datetime_format is None:
            self._datetime_format = "YYYY-MM-DD HH:MM:SS"
        else:
            self._datetime_format = datetime_format

        self._mode = mode

    @property
    def date_format(self) -> str:
        # [FUNCTION DOC]
        # Read only property: the format string used for date values.
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._date_format

    @property
    def datetime_format(self) -> str:
        # [FUNCTION DOC]
        # Read only property: the format string used for datetime values.
        # KNOWN ISSUE: the docstring is copied from the date format property and still
        #   describes dates with a date-only example, although this returns the datetime
        #   format.
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._datetime_format

    @property
    def if_sheet_exists(self) -> str:
        # [FUNCTION DOC]
        # Read only property: the policy (error, new, replace, overlay) applied when
        #   writing to an existing sheet in append mode.
        """
        How to behave when writing to a sheet that already exists in append mode.
        """
        return self._if_sheet_exists

    def __fspath__(self) -> str:
        # [FUNCTION DOC]
        # Path protocol support: returns the name attribute of the underlying handle, or
        #   an empty string if the handle has none.
        return getattr(self._handles.handle, "name", "")

    def _get_sheet_name(self, sheet_name: str | None) -> str:
        # [FUNCTION DOC]
        # HELPER: resolves an optional sheet name. Returns the given name, or the
        #   current sheet if none was given. Raises a value error if both are none.
        if sheet_name is None:
            sheet_name = self._cur_sheet
        if sheet_name is None:  # pragma: no cover
            raise ValueError("Must pass explicit sheet_name or set _cur_sheet property")
        return sheet_name

    def _value_with_fmt(
        self, val
    ) -> tuple[
        int | float | bool | str | datetime.datetime | datetime.date, str | None
    ]:
        # [FUNCTION DOC]
        # HELPER: converts a cell value to a type the engines can write and an optional
        #   number format. Input: any value. Returns a pair of the converted value and a
        #   format string or none. Integers, floats, booleans and decimals are converted
        #   to the plain Python type (no format). Datetime values get the datetime
        #   format and dates get the date format (datetime is checked first because it
        #   is a subclass of date). Time deltas become a fractional number of days with
        #   the general number format. Anything else is converted to text; if that text
        #   exceeds 32767 characters (the Excel cell limit) a user warning is emitted
        #   that the contents were truncated, though the returned text itself is not
        #   shortened here.
        """
        Convert numpy types to Python types for the Excel writers.

        Parameters
        ----------
        val : object
            Value to be written into cells

        Returns
        -------
        Tuple with the first element being the converted value and the second
            being an optional format
        """
        fmt = None

        if is_integer(val):
            val = int(val)
        elif is_float(val):
            val = float(val)
        elif is_bool(val):
            val = bool(val)
        elif is_decimal(val):
            val = Decimal(val)
        elif isinstance(val, datetime.datetime):
            fmt = self._datetime_format
        elif isinstance(val, datetime.date):
            fmt = self._date_format
        elif isinstance(val, datetime.timedelta):
            val = val.total_seconds() / 86400
            fmt = "0"
        else:
            val = str(val)
            # GH#56954
            # Excel's limitation on cell contents is 32767 characters
            # xref https://support.microsoft.com/en-au/office/excel-specifications-and-limits-1672b34d-7043-467e-8e27-269d656771c3
            if len(val) > 32767:
                warnings.warn(
                    f"Cell contents too long ({len(val)}), "
                    "truncated to 32767 characters",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
        return val, fmt

    @classmethod
    def check_extension(cls, ext: str) -> Literal[True]:
        # [FUNCTION DOC]
        # Class method that validates a file extension against the engine's supported
        #   extensions. Input: extension string, with or without a leading dot. Returns
        #   true if supported; otherwise raises a value error naming the engine and the
        #   extension.
        # KNOWN ISSUE: the docstring says an unsupported file type error is raised, but
        #   a value error is raised. The membership test is a substring check against
        #   each supported extension rather than an exact match, and the message reads
        #   the engine attribute from the class, which is a property object when
        #   accessed on the class rather than a string.
        """
        checks that path's extension against the Writer's supported
        extensions.  If it isn't supported, raises UnsupportedFiletypeError.
        """
        if ext.startswith("."):
            ext = ext[1:]
        if not any(ext in extension for extension in cls._supported_extensions):
            raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
        return True

    # Allow use as a contextmanager
    def __enter__(self) -> Self:
        # [FUNCTION DOC]
        # Context manager entry: returns the writer itself.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC]
        # Context manager exit: closes the writer (saving the workbook). Exception
        #   arguments are ignored and exceptions propagate.
        self.close()

    def close(self) -> None:
        # [FUNCTION DOC]
        # Save the workbook through the engine save hook and then close the underlying
        #   handles. Returns nothing. Acts as a file like close.
        """synonym for save, to make it more file-like"""
        self._save()
        self._handles.close()


XLS_SIGNATURES = (
    b"\x09\x00\x04\x00\x07\x00\x10\x00",  # BIFF2
    b"\x09\x02\x06\x00\x00\x00\x10\x00",  # BIFF3
    b"\x09\x04\x06\x00\x00\x00\x10\x00",  # BIFF4
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",  # Compound File Binary
)
ZIP_SIGNATURE = b"PK\x03\x04"
PEEK_SIZE = max(map(len, (*XLS_SIGNATURES, ZIP_SIGNATURE)))


def inspect_excel_format(
    content_or_path: FilePath | ReadBuffer[bytes],
    storage_options: StorageOptions | None = None,
) -> str | None:
    # [FUNCTION DOC]
    # Determine the format of an Excel file from its leading bytes and, if zipped, its
    #   member names. Inputs: content_or_path (path, URL or binary file like object) and
    #   storage_options for opening. Returns xls when the first bytes match one of the
    #   old binary signatures; none when the content is neither such a signature nor a
    #   zip signature; for zip archives inspects the lower cased member names
    #   (backslashes normalized to slashes) and returns xlsx if the workbook xml part
    #   exists, xlsb if the workbook binary part exists, ods if the content xml part
    #   exists, and otherwise zip. Raises a value error if reading yields nothing; the
    #   zip library raises its bad zip file error for a zip signature with invalid
    #   content.
    # Side effects: opens the input, rewinds the stream to the start before returning
    #   (and before inspecting the archive), and closes only handles it opened.
    """
    Inspect the path or content of an excel file and get its format.

    Adopted from xlrd: https://github.com/python-excel/xlrd.

    Parameters
    ----------
    content_or_path : str or file-like object
        Path to file or content of file to inspect. May be a URL.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

    Returns
    -------
    str or None
        Format of file if it can be determined.

    Raises
    ------
    ValueError
        If resulting stream is empty.
    BadZipFile
        If resulting stream does not have an XLS signature and is not a valid zipfile.
    """
    with get_handle(
        content_or_path, "rb", storage_options=storage_options, is_text=False
    ) as handle:
        stream = handle.handle
        stream.seek(0)
        buf = stream.read(PEEK_SIZE)
        if buf is None:
            raise ValueError("stream is empty")
        assert isinstance(buf, bytes)
        peek = buf
        stream.seek(0)

        if any(peek.startswith(sig) for sig in XLS_SIGNATURES):
            return "xls"
        elif not peek.startswith(ZIP_SIGNATURE):
            return None

        with zipfile.ZipFile(stream) as zf:
            # Workaround for some third party files that use forward slashes and
            # lower case names.
            component_names = {
                name.replace("\\", "/").lower() for name in zf.namelist()
            }

        if "xl/workbook.xml" in component_names:
            return "xlsx"
        if "xl/workbook.bin" in component_names:
            return "xlsb"
        if "content.xml" in component_names:
            return "ods"
        return "zip"


@set_module("pandas")
class ExcelFile:
    # [CLASS DOC]
    # Public wrapper around an Excel workbook that lets several sheets be parsed from
    #   one opened file. Not a subclass of any reader; it composes one engine reader.
    # State: engine (name string of the chosen engine), storage_options, _io (string
    #   form of the input), _reader (instance of the engine reader class created from
    #   the engine table), and the class level mapping of engine names to reader
    #   classes.
    # Supports use as a context manager. Key methods: parse, the book and sheet_names
    #   properties, close.
    """
    Class for parsing tabular Excel sheets into DataFrame objects.

    See read_excel for more documentation.

    Parameters
    ----------
    path_or_buffer : str, bytes, pathlib.Path,
        A file-like object, xlrd workbook or openpyxl workbook.
        If a string or path object, expected to be a path to a
        .xls, .xlsx, .xlsb, .xlsm, .odf, .ods, or .odt file.
    engine : str, default None
        If io is not a buffer or path, this must be set to identify io.
        Supported engines: ``xlrd``, ``openpyxl``, ``odf``, ``pyxlsb``, ``calamine``
        Engine compatibility :

        - ``xlrd`` supports old-style Excel files (.xls).
        - ``openpyxl`` supports newer Excel file formats.
        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
        - ``pyxlsb`` supports Binary Excel files.
        - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
          and OpenDocument (.ods) file formats.

        The engine `xlrd <https://xlrd.readthedocs.io/en/latest/>`_
        now only supports old-style ``.xls`` files.
        When ``engine=None``, the following logic will be
        used to determine the engine:

        - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
            then `odf <https://pypi.org/project/odfpy/>`_ will be used.
        - Otherwise if ``path_or_buffer`` is an xls format,
            ``xlrd`` will be used.
        - Otherwise if ``path_or_buffer`` is in xlsb format,
            `pyxlsb <https://pypi.org/project/pyxlsb/>`_ will be used.
        - Otherwise if `openpyxl <https://pypi.org/project/openpyxl/>`_ is installed,
            then ``openpyxl`` will be used.
        - Otherwise if ``xlrd >= 2.0`` is installed, a ``ValueError`` will be raised.

        .. warning::

           Please do not report issues when using ``xlrd`` to read ``.xlsx`` files.
           This is not supported, switch to using ``openpyxl`` instead.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.
    engine_kwargs : dict, optional
        Arbitrary keyword arguments passed to excel engine.

    See Also
    --------
    DataFrame.to_excel : Write DataFrame to an Excel file.
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Examples
    --------
    >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
    >>> with pd.ExcelFile("myfile.xls") as xls:  # doctest: +SKIP
    ...     df1 = pd.read_excel(xls, "Sheet1")  # doctest: +SKIP
    """

    from pandas.io.excel._calamine import CalamineReader
    from pandas.io.excel._odfreader import ODFReader
    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas.io.excel._pyxlsb import PyxlsbReader
    from pandas.io.excel._xlrd import XlrdReader

    _engines: Mapping[str, Any] = {
        "xlrd": XlrdReader,
        "openpyxl": OpenpyxlReader,
        "odf": ODFReader,
        "pyxlsb": PyxlsbReader,
        "calamine": CalamineReader,
    }

    def __init__(
        self,
        path_or_buffer,
        engine: str | None = None,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Create the file wrapper. Inputs: path_or_buffer (path, URL, buffer, engine
        #   workbook), engine (name or none for auto detection), storage_options,
        #   engine_kwargs (default none, treated as empty).
        # Behavior: raises a value error for an engine name that is not in the engine
        #   table. Stores the path as a string via the path conversion helper. When the
        #   engine is none: if the input is an xlrd workbook object (checked only when
        #   xlrd is installed, to avoid importing it needlessly) the format is xls;
        #   otherwise the format is sniffed from the content and a value error is raised
        #   when it cannot be determined. The engine is then taken from the reader
        #   configuration option for that format, and auto is resolved by the default
        #   engine helper. Finally constructs the engine reader with the string input,
        #   storage options and engine kwargs.
        # Side effects: opens the file through the reader.
        if engine_kwargs is None:
            engine_kwargs = {}

        if engine is not None and engine not in self._engines:
            raise ValueError(f"Unknown engine: {engine}")

        # Always a string
        self._io = stringify_path(path_or_buffer)

        if engine is None:
            # Only determine ext if it is needed
            ext: str | None = None

            if not isinstance(
                path_or_buffer, (str, os.PathLike, ExcelFile)
            ) and not is_file_like(path_or_buffer):
                # GH#56692 - avoid importing xlrd if possible
                if import_optional_dependency("xlrd", errors="ignore") is None:
                    xlrd_version = None
                else:
                    import xlrd

                    xlrd_version = Version(get_version(xlrd))

                if xlrd_version is not None and isinstance(path_or_buffer, xlrd.Book):
                    ext = "xls"

            if ext is None:
                ext = inspect_excel_format(
                    content_or_path=path_or_buffer, storage_options=storage_options
                )
                if ext is None:
                    raise ValueError(
                        "Excel file format cannot be determined, you must specify "
                        "an engine manually."
                    )

            engine = config.get_option(f"io.excel.{ext}.reader")
            if engine == "auto":
                engine = get_default_engine(ext, mode="reader")

        assert engine is not None
        self.engine = engine
        self.storage_options = storage_options

        self._reader = self._engines[engine](
            self._io,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    def __fspath__(self):
        # [FUNCTION DOC]
        # Path protocol support: returns the stored string form of the input.
        return self._io

    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        converters=None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values=None,
        parse_dates: list | dict | bool = False,
        date_format: str | dict[Hashable, str] | None = None,
        thousands: str | None = None,
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ) -> DataFrame | dict[str, DataFrame] | dict[int, DataFrame]:
        # [FUNCTION DOC]
        # Parse the selected sheet or sheets into tables. Same parameters as the reader
        #   function (sheet_name, header, names, index_col, usecols, converters,
        #   true_values, false_values, skiprows, nrows, na_values, parse_dates,
        #   date_format, thousands, comment, skipfooter, dtype_backend) plus extra
        #   keyword arguments such as dtype, decimal and verbose which are forwarded
        #   unchanged to the reader's parse method. Returns a single table for scalar
        #   sheet selection or a mapping of selector to table for a list or none
        #   selector.
        # KNOWN ISSUE: the signature does not list dtype, decimal or verbose and the
        #   docstring does not mention them, but the module level reader function passes
        #   them here and they only work because they travel through the extra keyword
        #   arguments.
        """
        Parse specified sheet(s) into a DataFrame.

        Equivalent to read_excel(ExcelFile, ...)  See the read_excel
        docstring for more info on accepted parameters.

        Parameters
        ----------
        sheet_name : str, int, list, or None, default 0
            Strings are used for sheet names. Integers are used in zero-indexed
            sheet positions (chart sheets do not count as a sheet position).
            Lists of strings/integers are used to request multiple sheets.
            When ``None``, will return a dictionary containing DataFrames for
            each sheet.
        header : int, list of int, default 0
            Row (0-indexed) to use for the column labels of the parsed
            DataFrame. If a list of integers is passed those row positions will
            be combined into a ``MultiIndex``. Use None if there is no header.
        names : array-like, default None
            List of column names to use. If file contains no header row,
            then you should explicitly pass header=None.
        index_col : int, str, list of int, default None
            Column (0-indexed) to use as the row labels of the DataFrame.
            Pass None if there is no such column.  If a list is passed,
            those columns will be combined into a ``MultiIndex``.  If a
            subset of data is selected with ``usecols``, index_col
            is based on the subset.

            Missing values will be forward filled to allow roundtripping with
            ``to_excel`` for ``merged_cells=True``. To avoid forward filling the
            missing values use ``set_index`` after reading the data instead of
            ``index_col``.
        usecols : str, list-like, or callable, default None
            * If None, then parse all columns.
            * If str, then indicates comma separated list of Excel column letters
              and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of
              both sides.
            * If list of int, then indicates list of column numbers to be parsed
              (0-indexed).
            * If list of string, then indicates list of column names to be parsed.
            * If callable, then evaluate each column name against it and parse the
              column if the callable returns ``True``.

            Returns a subset of the columns according to behavior above.
        converters : dict, default None
            Dict of functions for converting values in certain columns. Keys can
            either be integers or column labels, values are functions that take one
            input argument, the Excel cell content, and return the transformed
            content.
        true_values : list, default None
            Values to consider as True.
        false_values : list, default None
            Values to consider as False.
        skiprows : list-like, int, or callable, optional
            Line numbers to skip (0-indexed) or number of lines to skip (int) at the
            start of the file. If callable, the callable function will be evaluated
            against the row indices, returning True if the row should be skipped and
            False otherwise. An example of a valid callable argument would be ``lambda
            x: x in [0, 2]``.
        nrows : int, default None
            Number of rows to parse.
        na_values : scalar, str, list-like, or dict, default None
            Additional strings to recognize as NA/NaN. If dict passed, specific
            per-column NA values.
        parse_dates : bool, list-like, or dict, default False
            The behavior is as follows:

            * ``bool``. If True -> try parsing the index.
            * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
              each as a separate date column.
            * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and
              parse as a single date column.
            * ``dict``, e.g. {'foo' : [1, 3]} -> parse columns 1, 3 as date and call
              result 'foo'

            If a column or index contains an unparsable date, the entire column or
            index will be returned unaltered as an object data type. If you
            don`t want to parse some cells as date just change their type
            in Excel to "Text".For non-standard datetime parsing, use
            ``pd.to_datetime`` after ``pd.read_excel``.

            Note: A fast-path exists for iso8601-formatted dates.
        date_format : str or dict of column -> format, default ``None``
           If used in conjunction with ``parse_dates``, will parse dates
           according to this format. For anything more complex,
           please read in as ``object`` and then apply :func:`to_datetime` as-needed.
        thousands : str, default None
            Thousands separator for parsing string columns to numeric.  Note that
            this parameter is only necessary for columns stored as TEXT in Excel,
            any numeric columns will automatically be parsed, regardless of display
            format.
        comment : str, default None
            Comments out remainder of line. Pass a character or characters to this
            argument to indicate comments in the input file. Any data between the
            comment string and the end of the current line is ignored.
        skipfooter : int, default 0
            Rows at the end to skip (0-indexed).
        dtype_backend : {'numpy_nullable', 'pyarrow'}
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). If not specified, the default behavior
            is to not use nullable data types. If specified, the behavior
            is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
            * ``"pyarrow"``: returns pyarrow-backed nullable
              :class:`ArrowDtype` :class:`DataFrame`

            .. versionadded:: 2.0
        **kwds : dict, optional
            Arbitrary keyword arguments passed to excel engine.

        Returns
        -------
        DataFrame or dict of DataFrames
            DataFrame from the passed in Excel file.

        See Also
        --------
        read_excel : Read an Excel sheet values (xlsx) file into DataFrame.
        read_csv : Read a comma-separated values (csv) file into DataFrame.
        read_fwf : Read a table of fixed-width formatted lines into DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        >>> df.to_excel("myfile.xlsx")  # doctest: +SKIP
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.parse()  # doctest: +SKIP
        """
        return self._reader.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            parse_dates=parse_dates,
            date_format=date_format,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
            **kwds,
        )

    @property
    def book(self):
        # [FUNCTION DOC]
        # Read only property: the underlying engine workbook object, taken from the
        #   reader.
        """
        Gets the Excel workbook.

        Workbook is the top-level container for all document information.

        Returns
        -------
        Excel Workbook
            The workbook object of the type defined by the engine being used.

        See Also
        --------
        read_excel : Read an Excel file into a pandas DataFrame.

        Examples
        --------
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.book  # doctest: +SKIP
        <openpyxl.workbook.workbook.Workbook object at 0x11eb5ad70>
        >>> file.book.path  # doctest: +SKIP
        '/xl/workbook.xml'
        >>> file.book.active  # doctest: +SKIP
        <openpyxl.worksheet._read_only.ReadOnlyWorksheet object at 0x11eb5b370>
        >>> file.book.sheetnames  # doctest: +SKIP
        ['Sheet1', 'Sheet2']
        """
        return self._reader.book

    @property
    def sheet_names(self):
        # [FUNCTION DOC]
        # Read only property: the list of sheet names of the workbook, taken from the
        #   reader.
        """
        Names of the sheets in the document.

        This is particularly useful for loading a specific sheet into a DataFrame when
        you do not know the sheet names beforehand.

        Returns
        -------
        list of str
            List of sheet names in the document.

        See Also
        --------
        ExcelFile.parse : Parse a sheet into a DataFrame.
        read_excel : Read an Excel file into a pandas DataFrame. If you know the sheet
            names, it may be easier to specify them directly to read_excel.

        Examples
        --------
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.sheet_names  # doctest: +SKIP
        ["Sheet1", "Sheet2"]
        """
        return self._reader.sheet_names

    def close(self) -> None:
        # [FUNCTION DOC]
        # Close the reader and release the underlying file resources. Returns nothing.
        """close io if necessary"""
        self._reader.close()

    def __enter__(self) -> Self:
        # [FUNCTION DOC]
        # Context manager entry: returns the file wrapper itself.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC]
        # Context manager exit: closes the wrapper; exception arguments are ignored and
        #   exceptions propagate.
        self.close()
