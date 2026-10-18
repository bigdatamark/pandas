"""
Printing tools.
"""

# [MODULE DOC]
# Pretty printing tools used across pandas for converting objects to text and for
#   joining and justifying columns of text.
# Public contents: adjoin (glue lists of strings side by side), pprint_thing and
#   pprint_thing_encoded (convert any object, including nested sequences and mappings,
#   to text or bytes honoring display options), default_pprint, format_object_summary
#   (bracketed, truncated, line wrapped summary of a sequence as used in index and array
#   reprs), enable_data_resource_formatter (toggle the notebook table schema display),
#   PrettyDict (dictionary with abbreviated representation), the text adjustment classes
#   (plain and East Asian width aware) with their factory get_adjustment, and the type
#   alias for escape character specifications. The private helpers justify table cells
#   and print sequences and mappings.
# Relationships: reads display options from the global options; the console module
#   supplies the console width; the frame formatters in the sibling format module and
#   the string, html and info renderers use the adjustment objects and pprint_thing.
# Import-time side effects: defines two type variables and the escape characters type
#   alias.

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypeVar,
)
from unicodedata import east_asian_width

from pandas._config import get_option

from pandas.core.dtypes.inference import is_sequence

from pandas.io.formats.console import get_console_size

if TYPE_CHECKING:
    from pandas._typing import ListLike
EscapeChars: TypeAlias = Mapping[str, str] | Iterable[str]
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def adjoin(space: int, *lists: list[str], **kwargs: Any) -> str:
    # [FUNCTION DOC]
    # Glue several columns of text together side by side, separated by a number of
    #   spaces, producing a multi line string. Inputs: space (padding between columns),
    #   lists (one list of strings per column) and optional keyword arguments strlen
    #   (function measuring text width, default the built in length; needed for wide
    #   characters) and justfunc (function that justifies a list of strings to a width
    #   with a mode argument, default the plain left justify helper); the two are
    #   removed from the keyword arguments before use. Returns the lines joined by
    #   newlines.
    # Behavior: computes each column width as its widest cell plus the spacing, except
    #   the last column which gets only its widest cell (measured by plain length);
    #   shorter columns are padded at the top with blank cells so that all have the same
    #   number of rows; every column is left justified to its width; rows are
    #   concatenated and joined.
    # KNOWN ISSUE: the docstring lacks a Parameters heading (only an underline is
    #   present) and describes lists as a list of strings; the last column's width uses
    #   plain length rather than the supplied measuring function.
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    ----------
    space : int
        number of spaces for padding
    lists : str
        list of str which being joined
    strlen : callable
        function used to calculate the length of each str. Needed for unicode
        handling.
    justfunc : callable
        function used to justify str. Needed for unicode handling.
    """
    strlen = kwargs.pop("strlen", len)
    justfunc = kwargs.pop("justfunc", _adj_justify)

    newLists = []
    lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
    # not the last one
    lengths.append(max(map(len, lists[-1])))
    maxLen = max(map(len, lists))
    for i, lst in enumerate(lists):
        nl = justfunc(lst, lengths[i], mode="left")
        nl = ([" " * lengths[i]] * (maxLen - len(lst))) + nl
        newLists.append(nl)
    toJoin = zip(*newLists, strict=True)
    return "\n".join("".join(lines) for lines in toJoin)


def _adj_justify(texts: Iterable[str], max_len: int, mode: str = "right") -> list[str]:
    # [FUNCTION DOC]
    # HELPER: default justification function for the column joiner. Inputs: texts
    #   (iterable of strings), max_len (target width) and mode (left, center, or
    #   anything else meaning right; default right). Returns a new list of strings
    #   padded to the width in that mode.
    """
    Perform ljust, center, rjust against string or list-like
    """
    if mode == "left":
        return [x.ljust(max_len) for x in texts]
    elif mode == "center":
        return [x.center(max_len) for x in texts]
    else:
        return [x.rjust(max_len) for x in texts]


# Unicode consolidation
# ---------------------
#
# pprinting utility functions for generating Unicode text or
# bytes(3.x)/str(2.x) representations of objects.
# Try to use these as much as possible rather than rolling your own.
#
# When to use
# -----------
#
# 1) If you're writing code internal to pandas (no I/O directly involved),
#    use pprint_thing().
#
#    It will always return unicode text which can handled by other
#    parts of the package without breakage.
#
# 2) if you need to write something out to file, use
#    pprint_thing_encoded(encoding).
#
#    If no encoding is specified, it defaults to utf-8. Since encoding pure
#    ascii with utf-8 is a no-op you can safely use the default utf-8 if you're
#    working with straight ascii.


def _pprint_seq(
    seq: ListLike, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds: Any
) -> str:
    # [FUNCTION DOC]
    # HELPER: pretty printer for sequences, used by the main printer. Inputs: seq (list
    #   like), _nest_lvl (current nesting depth, internal), max_seq_items (limit of
    #   printed items; none uses the display option, then the full length; the value
    #   false means unlimited) and extra keyword arguments passed to the recursive
    #   printer for each element. Returns the text: sets use braces, frozen sets use the
    #   frozen set wrapper with braces, mutable sequences use square brackets and
    #   everything else round brackets. Items beyond the limit are replaced by a
    #   trailing ellipsis; a one element tuple gets a trailing comma.
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.

    bounds length of printed sequence, depending on options
    """
    if isinstance(seq, set):
        fmt = "{{{body}}}"
    elif isinstance(seq, frozenset):
        fmt = "frozenset({{{body}}})"
    else:
        fmt = "[{body}]" if hasattr(seq, "__setitem__") else "({body})"

    if max_seq_items is False:
        max_items = None
    else:
        max_items = max_seq_items or get_option("max_seq_items") or len(seq)

    s = iter(seq)
    # handle sets, no slicing
    r = []
    max_items_reached = False
    for i, item in enumerate(s):
        if (max_items is not None) and (i >= max_items):
            max_items_reached = True
            break
        r.append(pprint_thing(item, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds))
    body = ", ".join(r)

    if max_items_reached:
        body += ", ..."
    elif isinstance(seq, tuple) and len(seq) == 1:
        body += ","

    return fmt.format(body=body)


def _pprint_dict(
    seq: Mapping, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds: Any
) -> str:
    # [FUNCTION DOC]
    # HELPER: pretty printer for mappings, used by the main printer. Inputs: seq
    #   (mapping), _nest_lvl, max_seq_items (limit of printed pairs; none uses the
    #   display option, then the full length; false means unlimited) and extra keyword
    #   arguments passed to the printer for keys and values. Returns braces around comma
    #   separated key colon value pairs, followed by an ellipsis when truncated.
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.
    """
    fmt = "{{{things}}}"
    pairs = []

    pfmt = "{key}: {val}"

    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or get_option("max_seq_items") or len(seq)

    for k, v in list(seq.items())[:nitems]:
        pairs.append(
            pfmt.format(
                key=pprint_thing(k, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
                val=pprint_thing(v, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
            )
        )

    if nitems < len(seq):
        return fmt.format(things=", ".join(pairs) + ", ...")
    else:
        return fmt.format(things=", ".join(pairs))


def pprint_thing(
    thing: object,
    _nest_lvl: int = 0,
    escape_chars: EscapeChars | None = None,
    default_escapes: bool = False,
    quote_strings: bool = False,
    max_seq_items: int | None = None,
) -> str:
    # [FUNCTION DOC]
    # Convert any object to text. This is the sanctioned way of stringifying objects in
    #   pandas; it handles nested sequences and mappings. Inputs: thing (the object),
    #   _nest_lvl (internal recursion depth), escape_chars (characters to escape: a
    #   mapping from character to replacement, or an iterable of characters),
    #   default_escapes (for a mapping, whether it adds to or replaces the default
    #   escapes of tab, newline, carriage return and single quote; default false),
    #   quote_strings (wrap plain strings in single quotes; default false) and
    #   max_seq_items (limit for printed items; default none uses the display option).
    #   Returns the text.
    # Behavior: iterators (objects with a next method) are converted with plain string
    #   conversion; mappings and sequences (while the nesting depth is below the nest
    #   depth display option) are printed by the dedicated printers; a string with
    #   quoting enabled gets escaped and quoted; everything else is converted to text
    #   and has the escapes applied.
    # KNOWN ISSUE: the docstring says the default escapes flag controls whether the
    #   characters replace or add to the defaults, but it only has an effect when
    #   escape_chars is a mapping; for an iterable of characters each character must be
    #   one of the four default escape characters or a key error is raised. The mapping
    #   printer is called without the escape characters, so they are not applied inside
    #   mappings, and strings in mappings are always quoted.
    """
    This function is the sanctioned way of converting objects
    to a string representation and properly handles nested sequences.

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list[str] or Mapping[str, str], optional
        Characters to escape. If a Mapping is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : int or None, default None
        Pass through to other pretty printers to limit sequence printing

    Returns
    -------
    str
    """

    def as_escaped_string(
        thing: Any, escape_chars: EscapeChars | None = escape_chars
    ) -> str:
        # [FUNCTION DOC]
        # HELPER: nested function that converts an object to text and applies character
        #   escapes. Input: thing and escape_chars (defaults to the enclosing call's
        #   value). Builds a translation table from the default escapes; for a mapping,
        #   either merges it into the defaults or replaces them, and uses its keys as
        #   the characters to escape; otherwise uses the given iterable (or nothing).
        #   Returns the converted text with each escape character replaced by its
        #   translation.
        translate = {"\t": r"\t", "\n": r"\n", "\r": r"\r", "'": r"\'"}
        if isinstance(escape_chars, Mapping):
            if default_escapes:
                translate.update(escape_chars)
            else:
                translate = escape_chars  # type: ignore[assignment]
            escape_chars = list(escape_chars.keys())
        else:
            escape_chars = escape_chars or ()

        result = str(thing)
        for c in escape_chars:
            result = result.replace(c, translate[c])
        return result

    if hasattr(thing, "__next__"):
        return str(thing)
    elif isinstance(thing, Mapping) and _nest_lvl < get_option(
        "display.pprint_nest_depth"
    ):
        result = _pprint_dict(
            thing, _nest_lvl, quote_strings=True, max_seq_items=max_seq_items
        )
    elif is_sequence(thing) and _nest_lvl < get_option("display.pprint_nest_depth"):
        result = _pprint_seq(
            # error: Argument 1 to "_pprint_seq" has incompatible type "object";
            # expected "ExtensionArray | ndarray[Any, Any] | Index | Series |
            # SequenceNotStr[Any] | range"
            thing,  # type: ignore[arg-type]
            _nest_lvl,
            escape_chars=escape_chars,
            quote_strings=quote_strings,
            max_seq_items=max_seq_items,
        )
    elif isinstance(thing, str) and quote_strings:
        result = f"'{as_escaped_string(thing)}'"
    else:
        result = as_escaped_string(thing)

    return result


def pprint_thing_encoded(
    object: object, encoding: str = "utf-8", errors: str = "replace"
) -> bytes:
    # [FUNCTION DOC]
    # Convert an object to text with the main printer and encode it to bytes. Inputs:
    #   object, encoding (default utf-8) and errors (error handling for encoding;
    #   default replace). Returns the bytes.
    value = pprint_thing(object)  # get unicode representation of object
    return value.encode(encoding, errors)


def enable_data_resource_formatter(enable: bool) -> None:
    # [FUNCTION DOC]
    # Turn the notebook table schema display format on or off. Input: enable (true to
    #   register and enable the data resource formatter, false to disable it). Does
    #   nothing when IPython has not been imported or there is no active shell. When
    #   enabling and the formatter for the data resource media type is not yet
    #   registered, defines and registers a formatter class that uses the table schema
    #   representation method of objects; then enables it. When disabling, disables the
    #   formatter if it is registered. Side effects: modifies the active IPython display
    #   formatters.
    if "IPython" not in sys.modules:
        # definitely not in IPython
        return
    from IPython import get_ipython

    # error: Call to untyped function "get_ipython" in typed context
    ip = get_ipython()
    if ip is None:
        # still not in IPython
        return

    formatters = ip.display_formatter.formatters
    mimetype = "application/vnd.dataresource+json"

    if enable:
        if mimetype not in formatters:
            # define tableschema formatter
            from IPython.core.formatters import BaseFormatter
            from traitlets import ObjectName

            class TableSchemaFormatter(BaseFormatter):
                # [CLASS DOC]
                # HELPER: formatter class defined on demand inside the enable function
                #   and registered with IPython for the data resource media type. It
                #   subclasses the IPython base formatter, uses the table schema
                #   representation method of objects as its print method and accepts
                #   only dictionaries as results.
                print_method = ObjectName("_repr_data_resource_")
                _return_type = (dict,)

            # register it:
            formatters[mimetype] = TableSchemaFormatter()
        # enable it if it's been disabled:
        formatters[mimetype].enabled = True
    # unregister tableschema mime-type
    elif mimetype in formatters:
        formatters[mimetype].enabled = False


def default_pprint(thing: Any, max_seq_items: int | None = None) -> str:
    # [FUNCTION DOC]
    # Print an object using the main printer with tabs, carriage returns and newlines
    #   escaped and strings quoted. Inputs: thing and max_seq_items (default none).
    #   Returns the text.
    return pprint_thing(
        thing,
        escape_chars=("\t", "\r", "\n"),
        quote_strings=True,
        max_seq_items=max_seq_items,
    )


def format_object_summary(
    obj: ListLike,
    formatter: Callable,
    is_justify: bool = True,
    name: str | None = None,
    indent_for_name: bool = True,
    line_break_each_value: bool = False,
) -> str:
    # [FUNCTION DOC]
    # Build the bracketed summary text of a sequence, as used by index and array
    #   representations. Inputs: obj (sized, indexable sequence), formatter (function
    #   formatting one element to text), is_justify (right justify values to a common
    #   width when needed; default true), name (label whose length determines the
    #   continuation indent; default the class name of obj), indent_for_name (indent
    #   continuation lines to line up with the name; default true) and
    #   line_break_each_value (put every value, each itself a sequence, on its own line;
    #   default false). Returns the summary text.
    # Behavior: gets the display width from the console size, falling back to the width
    #   option or 80. Reads the maximum items option; truncation applies when the length
    #   exceeds it. Empty, one element and two element inputs (when not breaking per
    #   value) are printed inline. Otherwise shows leading and trailing elements (at
    #   most ten each side, half the limit each; a limit of one shows only the last
    #   element), formats them, optionally justifies them (to a common width when
    #   truncated or too wide for a line; per position when breaking each value), packs
    #   the values into lines no wider than the display width using separators, inserts
    #   an ellipsis line when truncated, closes the bracket and adds a trailing
    #   separator or line break depending on whether it fits on one row. In per value
    #   mode each value is shortened to the number of items that fits the available
    #   width.
    """
    Return the formatted obj as a unicode string

    Parameters
    ----------
    obj : object
        must be iterable and support __getitem__
    formatter : callable
        string formatter for an element
    is_justify : bool
        should justify the display
    name : name, optional
        defaults to the class name of the obj
    indent_for_name : bool, default True
        Whether subsequent lines should be indented to
        align with the name.
    line_break_each_value : bool, default False
        If True, inserts a line break for each value of ``obj``.
        If False, only break lines when the a line of values gets wider
        than the display width.

    Returns
    -------
    summary string
    """
    display_width, _ = get_console_size()
    if display_width is None:
        display_width = get_option("display.width") or 80
    if name is None:
        name = type(obj).__name__

    if indent_for_name:
        name_len = len(name)
        space1 = f"\n{(' ' * (name_len + 1))}"
        space2 = f"\n{(' ' * (name_len + 2))}"
    else:
        space1 = "\n"
        space2 = "\n "  # space for the opening '['

    n = len(obj)
    if line_break_each_value:
        # If we want to vertically align on each value of obj, we need to
        # separate values by a line break and indent the values
        sep = ",\n " + " " * len(name)
    else:
        sep = ","
    max_seq_items = get_option("display.max_seq_items") or n

    # are we a truncated display
    is_truncated = n > max_seq_items

    # adj can optionally handle unicode eastern asian width
    adj = get_adjustment()

    def _extend_line(
        s: str, line: str, value: str, display_width: int, next_line_prefix: str
    ) -> tuple[str, str]:
        # [FUNCTION DOC]
        # HELPER: nested function that appends a value to the current line. Inputs: s
        #   (text accumulated so far), line (current line), value (text to append),
        #   display_width and next_line_prefix. If the current line plus the value would
        #   reach the display width, the current line (right stripped) is flushed to s
        #   and a new line starting with the prefix is begun. Returns the pair of
        #   updated accumulated text and current line.
        if adj.len(line.rstrip()) + adj.len(value.rstrip()) >= display_width:
            s += line.rstrip()
            line = next_line_prefix
        line += value
        return s, line

    def best_len(values: list[str]) -> int:
        # [FUNCTION DOC]
        # HELPER: nested function returning the largest display width among the given
        #   strings, or zero for an empty list.
        if values:
            return max(adj.len(x) for x in values)
        else:
            return 0

    close = ", "

    if n == 0:
        summary = f"[]{close}"
    elif n == 1 and not line_break_each_value:
        first = formatter(obj[0])
        summary = f"[{first}]{close}"
    elif n == 2 and not line_break_each_value:
        first = formatter(obj[0])
        last = formatter(obj[-1])
        summary = f"[{first}, {last}]{close}"
    else:
        if max_seq_items == 1:
            # If max_seq_items=1 show only last element
            head = []
            tail = [formatter(x) for x in obj[-1:]]
        elif n > max_seq_items:
            n = min(max_seq_items // 2, 10)
            head = [formatter(x) for x in obj[:n]]
            tail = [formatter(x) for x in obj[-n:]]
        else:
            head = []
            tail = [formatter(x) for x in obj]

        # adjust all values to max length if needed
        if is_justify:
            if line_break_each_value:
                # Justify each string in the values of head and tail, so the
                # strings will right align when head and tail are stacked
                # vertically.
                head, tail = _justify(head, tail)
            elif is_truncated or not (
                len(", ".join(head)) < display_width
                and len(", ".join(tail)) < display_width
            ):
                # Each string in head and tail should align with each other
                max_length = max(best_len(head), best_len(tail))
                head = [x.rjust(max_length) for x in head]
                tail = [x.rjust(max_length) for x in tail]
            # If we are not truncated and we are only a single
            # line, then don't justify

        if line_break_each_value:
            # Now head and tail are of type List[Tuple[str]]. Below we
            # convert them into List[str], so there will be one string per
            # value. Also truncate items horizontally if wider than
            # max_space
            max_space = display_width - len(space2)
            value = tail[0]
            max_items = 1
            for num_items in reversed(range(1, len(value) + 1)):
                pprinted_seq = _pprint_seq(value, max_seq_items=num_items)
                if len(pprinted_seq) < max_space:
                    max_items = num_items
                    break
            head = [_pprint_seq(x, max_seq_items=max_items) for x in head]
            tail = [_pprint_seq(x, max_seq_items=max_items) for x in tail]

        summary = ""
        line = space2

        for head_value in head:
            word = head_value + sep + " "
            summary, line = _extend_line(summary, line, word, display_width, space2)

        if is_truncated:
            # remove trailing space of last line
            summary += line.rstrip() + space2 + "..."
            line = space2

        for tail_item in tail[:-1]:
            word = tail_item + sep + " "
            summary, line = _extend_line(summary, line, word, display_width, space2)

        # last value: no sep added + 1 space of width used for trailing ','
        summary, line = _extend_line(summary, line, tail[-1], display_width - 2, space2)
        summary += line

        # right now close is either '' or ', '
        # Now we want to include the ']', but not the maybe space.
        close = "]" + close.rstrip(" ")
        summary += close

        if len(summary) > (display_width) or line_break_each_value:
            summary += space1
        else:  # one row
            summary += " "

        # remove initial space
        summary = "[" + summary[len(space2) :]

    return summary


def _justify(
    head: list[Sequence[str]], tail: list[Sequence[str]]
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    # [FUNCTION DOC]
    # HELPER: right justifies items so that they line up when the head and tail lists
    #   are stacked vertically. Inputs: head and tail, lists of sequences of strings.
    #   For each position within the inner sequences finds the widest string across both
    #   lists and right justifies to it. Returns the pair of head and tail as lists of
    #   tuples of strings.
    """
    Justify items in head and tail, so they are right-aligned when stacked.

    Parameters
    ----------
    head : list-like of list-likes of strings
    tail : list-like of list-likes of strings

    Returns
    -------
    tuple of list of tuples of strings
        Same as head and tail, but items are right aligned when stacked
        vertically.

    Examples
    --------
    >>> _justify([["a", "b"]], [["abc", "abcd"]])
    ([('  a', '   b')], [('abc', 'abcd')])
    """
    combined = head + tail

    # For each position for the sequences in ``combined``,
    # find the length of the largest string.
    max_length = [0] * len(combined[0])
    for inner_seq in combined:
        length = [len(item) for item in inner_seq]
        max_length = [max(x, y) for x, y in zip(max_length, length, strict=True)]

    # justify each item in each list-like in head and tail using max_length
    head_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length, strict=True))
        for seq in head
    ]
    tail_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length, strict=True))
        for seq in tail
    ]
    return head_tuples, tail_tuples


class PrettyDict(dict[_KT, _VT]):
    # [CLASS DOC]
    # Dictionary subclass whose representation is produced by the main printer, so it is
    #   abbreviated according to the display options. Generic over key and value types;
    #   no extra state.
    """Dict extension to support abbreviated __repr__"""

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns the text produced by the main printer for this dictionary (limited
        #   number of items, nested printing).
        return pprint_thing(self)


class _TextAdjustment:
    # [CLASS DOC]
    # Default text measurement and justification object used by the frame formatters.
    #   State: encoding (the display encoding option). Methods: len (character count),
    #   justify (pad strings to a width in a mode) and adjoin (the module level column
    #   joiner configured with this object's measuring and justify methods). Subclassed
    #   to account for East Asian wide characters.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Store the display encoding option.
        self.encoding = get_option("display.encoding")

    def len(self, text: str) -> int:
        # [FUNCTION DOC]
        # Returns the number of characters in the text.
        return len(text)

    def justify(self, texts: Any, max_len: int, mode: str = "right") -> list[str]:
        # [FUNCTION DOC]
        # Pad each string of an iterable to a width. Inputs: texts, max_len and mode
        #   (left, center, otherwise right; default right). Returns a new list of padded
        #   strings.
        """
        Perform ljust, center, rjust against string or list-like
        """
        if mode == "left":
            return [x.ljust(max_len) for x in texts]
        elif mode == "center":
            return [x.center(max_len) for x in texts]
        else:
            return [x.rjust(max_len) for x in texts]

    def adjoin(self, space: int, *lists: Any, **kwargs: Any) -> str:
        # [FUNCTION DOC]
        # Join columns side by side using the module level joiner with this object's
        #   length and justify methods. Inputs: space, the lists of strings and extra
        #   keyword arguments. Returns the joined multi line text.
        return adjoin(space, *lists, strlen=self.len, justfunc=self.justify, **kwargs)


class _EastAsianTextAdjustment(_TextAdjustment):
    # [CLASS DOC]
    # Text adjustment that measures width in terminal cells, counting East Asian wide
    #   and fullwidth characters as two. State: ambiguous_width (two if the ambiguous
    #   characters as wide option is set, else one) and the table mapping East Asian
    #   width classes (narrow, neutral, wide, fullwidth, halfwidth) to cell counts.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialize the base class, set the ambiguous width from the display option and
        #   define the table mapping the Unicode East Asian width class to a number of
        #   terminal cells.
        super().__init__()
        if get_option("display.unicode.ambiguous_as_wide"):
            self.ambiguous_width = 2
        else:
            self.ambiguous_width = 1

        # Definition of East Asian Width
        # https://unicode.org/reports/tr11/
        # Ambiguous width can be changed by option
        self._EAW_MAP = {"Na": 1, "N": 1, "W": 2, "F": 2, "H": 1}

    def len(self, text: str) -> int:
        # [FUNCTION DOC]
        # Returns the display width of the text: the sum of per character widths based
        #   on the Unicode East Asian width class (ambiguous characters use the
        #   configured width). Non string input is measured with the plain length.
        """
        Calculate display width considering unicode East Asian Width
        """
        if not isinstance(text, str):
            return len(text)

        return sum(
            self._EAW_MAP.get(east_asian_width(c), self.ambiguous_width) for c in text
        )

    def justify(
        self, texts: Iterable[str], max_len: int, mode: str = "right"
    ) -> list[str]:
        # [FUNCTION DOC]
        # Pad each string to a display width. Inputs: texts, max_len and mode (left,
        #   center, otherwise right; default right). Because widths differ from
        #   character counts, the padding for each string is its target width minus its
        #   display width plus its character count. Returns a new list of padded
        #   strings.
        # re-calculate padding space per str considering East Asian Width
        def _get_pad(t: str) -> int:
            # [FUNCTION DOC]
            # HELPER: nested function computing the pad width for one text so that the
            #   padded text reaches the target display width: the target minus the
            #   display width plus the character count.
            return max_len - self.len(t) + len(t)

        if mode == "left":
            return [x.ljust(_get_pad(x)) for x in texts]
        elif mode == "center":
            return [x.center(_get_pad(x)) for x in texts]
        else:
            return [x.rjust(_get_pad(x)) for x in texts]


def get_adjustment() -> _TextAdjustment:
    # [FUNCTION DOC]
    # Factory returning the text adjustment object chosen by the East Asian width
    #   display option: the width aware one when the option is set, otherwise the plain
    #   one.
    use_east_asian_width = get_option("display.unicode.east_asian_width")
    if use_east_asian_width:
        return _EastAsianTextAdjustment()
    else:
        return _TextAdjustment()
