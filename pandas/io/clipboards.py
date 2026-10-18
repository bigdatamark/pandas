"""io on the clipboard"""

# [MODULE DOC]
# Clipboard input/output for pandas objects. Public contents: the read
# function (exposed as the top-level clipboard reader) and the write helper
# used by the frame/series clipboard methods. Both delegate to the vendored
# clipboard backend package for raw text access and to the csv reader or the
# object's csv writer for parsing/formatting. Only utf-8 text is supported.
# Import-time side effects: none (the backend is imported lazily in the
# functions).

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING
import warnings

from pandas._libs import lib
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.generic import ABCDataFrame

from pandas import (
    get_option,
    option_context,
)

if TYPE_CHECKING:
    from pandas._typing import DtypeBackend


@set_module("pandas")
def read_clipboard(
    sep: str = r"\s+",
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    **kwargs,
):  # pragma: no cover
    # [FUNCTION DOC]
    # Read the system clipboard text and parse it into a data frame via the
    # csv reader.
    # Inputs: sep is the delimiter string or regex (default any whitespace run);
    # dtype_backend selects numpy-nullable or pyarrow backed dtypes (validated,
    # default means not specified); extra keyword arguments are forwarded to the
    # csv reader. An encoding keyword is removed from the arguments first
    # (default utf-8).
    # Behavior: obtains text from the clipboard, decodes it if it is bytes,
    # then inspects up to the first ten lines (from at most the first 10000
    # characters, dropping the last fragment after the final newline). If more
    # than one line and every line has the same non-zero count of tabs (after
    # stripping leading spaces), it assumes an Excel copy: the separator becomes
    # tab and, if the first line has leading whitespace, the index column
    # keyword defaults to that many leading columns. Otherwise sep must be a
    # string. A multi-character separator selects the python engine when no
    # engine was given, and warns if the c engine was requested explicitly.
    # Returns the parsed data frame.
    # Raises: not-implemented error if the encoding is not utf-8 (case and
    # hyphen insensitive); value error if sep is not a string (and not an Excel
    # copy).
    # Side effects: reads the clipboard; may emit a user warning.
    # KNOWN ISSUE: the decode step reads the encoding from the keyword
    # arguments after it has been removed from them, so it always falls back to
    # the display encoding option rather than the validated encoding.
    r"""
    Read text from clipboard and pass to :func:`~pandas.read_csv`.

    Parses clipboard contents similar to how CSV files are parsed
    using :func:`~pandas.read_csv`.

    Parameters
    ----------
    sep : str, default '\\s+'
        A string or regex delimiter. The default of ``'\\s+'`` denotes
        one or more whitespace characters.

    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    **kwargs
        See :func:`~pandas.read_csv` for the full argument list.

    Returns
    -------
    DataFrame
        A parsed :class:`~pandas.DataFrame` object.

    See Also
    --------
    DataFrame.to_clipboard : Copy object to the system clipboard.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
    >>> df.to_clipboard()  # doctest: +SKIP
    >>> pd.read_clipboard()  # doctest: +SKIP
         A  B  C
    0    1  2  3
    1    4  5  6
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # only utf-8 is valid for passed value because that's what clipboard
    # supports
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise NotImplementedError("reading from clipboard only supports utf-8 encoding")

    check_dtype_backend(dtype_backend)

    from pandas.io.clipboard import clipboard_get
    from pandas.io.parsers import read_csv

    text = clipboard_get()

    # Try to decode (if needed, as "text" might already be a string here).
    try:
        text = text.decode(kwargs.get("encoding") or get_option("display.encoding"))
    except AttributeError:
        pass

    # Excel copies into clipboard with \t separation
    # inspect no more then the 10 first lines, if they
    # all contain an equal number (>0) of tabs, infer
    # that this came from excel and set 'sep' accordingly
    lines = text[:10000].split("\n")[:-1][:10]

    # Need to remove leading white space, since read_csv
    # accepts:
    #    a  b
    # 0  1  2
    # 1  3  4

    counts = {x.lstrip(" ").count("\t") for x in lines}
    if len(lines) > 1 and len(counts) == 1 and counts.pop() != 0:
        sep = "\t"
        # check the number of leading tabs in the first line
        # to account for index columns
        index_length = len(lines[0]) - len(lines[0].lstrip(" \t"))
        if index_length != 0:
            kwargs.setdefault("index_col", list(range(index_length)))

    elif not isinstance(sep, str):
        raise ValueError(f"{sep=} must be a string")

    # Regex separator currently only works with python engine.
    # Default to python if separator is multi-character (regex)
    if len(sep) > 1 and kwargs.get("engine") is None:
        kwargs["engine"] = "python"
    elif len(sep) > 1 and kwargs.get("engine") == "c":
        warnings.warn(
            "read_clipboard with regex separator does not work properly with c engine.",
            stacklevel=find_stack_level(),
        )

    return read_csv(StringIO(text), sep=sep, dtype_backend=dtype_backend, **kwargs)


def to_clipboard(
    obj, excel: bool | None = True, sep: str | None = None, **kwargs
) -> None:  # pragma: no cover
    # [FUNCTION DOC]
    # Write an object's text representation to the system clipboard.
    # Inputs: obj is the data frame or series (anything with a csv writer);
    # excel (default true; none is treated as true) selects csv output with a
    # separator suitable for pasting into spreadsheets; sep is the separator
    # (default tab, only used in excel mode); other keywords go to the csv
    # writer (or the string renderer in non-excel mode). An encoding keyword is
    # removed first (default utf-8).
    # Behavior: in excel mode writes csv text to a buffer using utf-8 and
    # puts it on the clipboard; if the writer raises a type error (for example
    # a multi-character separator) it warns and falls through to the plain
    # path. With excel false and a sep given it warns that sep is ignored. The
    # plain path renders a data frame as a string with unlimited column width
    # (temporarily changing the display option) or any other object via string
    # conversion, and copies that.
    # Raises: value error if the encoding is not utf-8.
    # Returns nothing; side effects: writes the clipboard, may warn.
    """
    Attempt to write text representation of object to the system clipboard
    The clipboard can be then pasted into Excel for example.

    Parameters
    ----------
    obj : the object to write to the clipboard
    excel : bool, defaults to True
            if True, use the provided separator, writing in a csv
            format for allowing easy pasting into excel.
            if False, write a string representation of the object
            to the clipboard
    sep : optional, defaults to tab
    other keywords are passed to to_csv

    Notes
    -----
    Requirements for your platform
      - Linux: xclip, or xsel (with PyQt4 modules)
      - Windows:
      - OS X:
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # testing if an invalid encoding is passed to clipboard
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise ValueError("clipboard only supports utf-8 encoding")

    from pandas.io.clipboard import clipboard_set

    if excel is None:
        excel = True

    if excel:
        try:
            if sep is None:
                sep = "\t"
            buf = StringIO()

            # clipboard_set (pyperclip) expects unicode
            obj.to_csv(buf, sep=sep, encoding="utf-8", **kwargs)
            text = buf.getvalue()

            clipboard_set(text)
            return
        except TypeError:
            warnings.warn(
                "to_clipboard in excel mode requires a single character separator.",
                stacklevel=find_stack_level(),
            )
    elif sep is not None:
        warnings.warn(
            "to_clipboard with excel=False ignores the sep argument.",
            stacklevel=find_stack_level(),
        )

    if isinstance(obj, ABCDataFrame):
        # str(df) has various unhelpful defaults, like truncation
        with option_context("display.max_colwidth", None):
            objstr = obj.to_string(**kwargs)
    else:
        objstr = str(obj)
    clipboard_set(objstr)
