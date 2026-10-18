"""
:func:`~pandas.eval` source string parsing functions
"""

# [MODULE DOC]
# Tokenizing support for expression strings that contain backtick-quoted
# column names (names that are not valid identifiers). Public contents:
# create_valid_python_identifier, clean_backtick_quoted_toks, clean_column_name,
# tokenize_string, the parse-state enumeration and the private splitter. Also
# defines the token number constant reserved for backtick-quoted strings.
# Used by the expr module and by code that looks up columns referred to via
# backticks. No import-time side effects.

from __future__ import annotations

from enum import Enum
from io import StringIO
from keyword import iskeyword
import token
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

# A token value Python's tokenizer probably will never use.
BACKTICK_QUOTED_STRING = 100


def create_valid_python_identifier(name: str) -> str:
    # [FUNCTION DOC]
    # Converts an arbitrary string into a valid Python identifier.
    # Input name: any string. If it is already a non-keyword identifier it is
    # returned unchanged. Otherwise non-ASCII characters are first replaced by
    # backslash-escape sequences (backslashes then become either a unicode
    # marker or a backslash marker), then each operator or punctuation
    # character is replaced by an underscore-wrapped token name (using the
    # tokenizer's exact-token table plus extras for space, question mark,
    # exclamation mark, dollar, euro, degree, quotes, hash and backtick), and a
    # fixed prefix is added. Raises a syntax error if the result is still not a
    # valid identifier.
    """
    Create valid Python identifiers from any string.

    Check if name contains any special characters. If it contains any
    special characters, the special characters will be replaced by
    a special string and a prefix is added.

    Raises
    ------
    SyntaxError
        If the returned name is not a Python valid identifier, raise an exception.
    """
    if name.isidentifier() and not iskeyword(name):
        return name

    # Escape characters that fall outside the ASCII range (U+0001..U+007F).
    # GH 49633
    gen = (
        (c, "".join(chr(b) for b in c.encode("ascii", "backslashreplace")))
        for c in name
    )
    name = "".join(
        c_escaped.replace("\\", "_UNICODE_" if c != c_escaped else "_BACKSLASH_")
        for c, c_escaped in gen
    )

    # Create a dict with the special characters and their replacement string.
    # EXACT_TOKEN_TYPES contains these special characters
    # token.tok_name contains a readable description of the replacement string.
    special_characters_replacements = {
        char: f"_{token.tok_name[tokval]}_"
        for char, tokval in (tokenize.EXACT_TOKEN_TYPES.items())
    }
    special_characters_replacements.update(
        {
            " ": "_",
            "?": "_QUESTIONMARK_",
            "!": "_EXCLAMATIONMARK_",
            "$": "_DOLLARSIGN_",
            "€": "_EUROSIGN_",
            "°": "_DEGREESIGN_",
            "'": "_SINGLEQUOTE_",
            '"': "_DOUBLEQUOTE_",
            "#": "_HASH_",
            "`": "_BACKTICK_",
        }
    )

    name = "".join([special_characters_replacements.get(char, char) for char in name])
    name = f"BACKTICK_QUOTED_STRING_{name}"

    if not name.isidentifier():
        raise SyntaxError(f"Could not convert '{name}' to a valid Python identifier.")

    return name


def clean_backtick_quoted_toks(tok: tuple[int, str]) -> tuple[int, str]:
    # [FUNCTION DOC]
    # Maps a (token number, token text) pair: if the number is the reserved
    # backtick-quoted number, returns the name token type with the text
    # converted to a valid identifier; otherwise returns the pair unchanged.
    # KNOWN ISSUE: the docstring refers to a function named with a leading
    # underscore, but the actual function has no underscore. Also, the
    # tokenizer in this module already replaces backtick-quoted text by
    # identifiers before tokenizing, so the reserved number is never produced
    # here and this mapping is effectively a pass-through.
    """
    Clean up a column name if surrounded by backticks.

    Backtick quoted string are indicated by a certain tokval value. If a string
    is a backtick quoted token it will processed by
    :func:`_create_valid_python_identifier` so that the parser can find this
    string when the query is executed.
    In this case the tok will get the NAME tokval.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tok : Tuple[int, str]
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    if toknum == BACKTICK_QUOTED_STRING:
        return tokenize.NAME, create_valid_python_identifier(tokval)
    return toknum, tokval


def clean_column_name(name: Hashable) -> Hashable:
    # [FUNCTION DOC]
    # Predicts the identifier a column label would receive when referenced via
    # backticks in an expression. Input name: any hashable label. String labels
    # have backticks doubled, then the label is wrapped in backticks,
    # tokenized, and the first token's text is converted to a valid identifier.
    # If a syntax error occurs, returns name (the original, or the
    # backtick-doubled form for strings) unchanged.
    """
    Function to emulate the cleaning of a backtick quoted name.

    The purpose for this function is to see what happens to the name of
    identifier if it goes to the process of being parsed a Python code
    inside a backtick quoted string and than being cleaned
    (removed of any special characters).

    Parameters
    ----------
    name : hashable
        Name to be cleaned.

    Returns
    -------
    name : hashable
        Returns the name after tokenizing and cleaning.
    """
    try:
        # Escape backticks
        name = name.replace("`", "``") if isinstance(name, str) else name

        tokenized = tokenize_string(f"`{name}`")
        tokval = next(tokenized)[1]
        return create_valid_python_identifier(tokval)
    except SyntaxError:
        return name


class ParseState(Enum):
    # [CLASS DOC]
    # HELPER: enumeration of lexical states used by the backtick splitter:
    # default (outside any quote), inside backticks, inside single quotes and
    # inside double quotes.
    DEFAULT = 0
    IN_BACKTICK = 1
    IN_SINGLE_QUOTE = 2
    IN_DOUBLE_QUOTE = 3


def _split_by_backtick(s: str) -> list[tuple[bool, str]]:
    # [FUNCTION DOC]
    # HELPER: splits source text into backtick-quoted and plain segments for
    # the tokenizer. Input s: source string. Returns a list of pairs (is
    # backtick-quoted, text); quoted segments include their surrounding
    # backticks and a doubled backtick inside is treated as an escaped
    # backtick. Backticks inside single or double quoted strings are ignored
    # (a quote preceded by a backslash does not close the string). An
    # unterminated backtick segment is emitted as plain text at the end.
    """
    Splits a str into substrings along backtick characters (`).

    Disregards backticks inside quotes.

    Parameters
    ----------
    s : str
        The Python source code string.

    Returns
    -------
    substrings: list[tuple[bool, str]]
        List of tuples, where each tuple has two elements:
        The first is a boolean indicating if the substring is backtick-quoted.
        The second is the actual substring.
    """
    substrings = []
    substr: list[str] = []  # Will join into a string before adding to `substrings`
    i = 0
    parse_state = ParseState.DEFAULT
    while i < len(s):
        char = s[i]

        match char:
            case "`":
                # start of a backtick-quoted string
                if parse_state == ParseState.DEFAULT:
                    if substr:
                        substrings.append((False, "".join(substr)))

                    substr = [char]
                    i += 1
                    parse_state = ParseState.IN_BACKTICK
                    continue

                elif parse_state == ParseState.IN_BACKTICK:
                    # escaped backtick inside a backtick-quoted string
                    next_char = s[i + 1] if (i != len(s) - 1) else None
                    if next_char == "`":
                        substr.append(char)
                        substr.append(next_char)
                        i += 2
                        continue

                    # end of the backtick-quoted string
                    else:
                        substr.append(char)
                        substrings.append((True, "".join(substr)))

                        substr = []
                        i += 1
                        parse_state = ParseState.DEFAULT
                        continue
            case "'":
                # start of a single-quoted string
                if parse_state == ParseState.DEFAULT:
                    parse_state = ParseState.IN_SINGLE_QUOTE
                # end of a single-quoted string
                elif (parse_state == ParseState.IN_SINGLE_QUOTE) and (s[i - 1] != "\\"):
                    parse_state = ParseState.DEFAULT
            case '"':
                # start of a double-quoted string
                if parse_state == ParseState.DEFAULT:
                    parse_state = ParseState.IN_DOUBLE_QUOTE
                # end of a double-quoted string
                elif (parse_state == ParseState.IN_DOUBLE_QUOTE) and (s[i - 1] != "\\"):
                    parse_state = ParseState.DEFAULT
        substr.append(char)
        i += 1

    if substr:
        substrings.append((False, "".join(substr)))

    return substrings


def tokenize_string(source: str) -> Iterator[tuple[int, str]]:
    # [FUNCTION DOC]
    # Generator that tokenizes expression source. Input source: string that
    # may contain backtick-quoted names. Each quoted segment is first replaced
    # by the valid identifier for its inner text (may raise a syntax error),
    # then the standard Python tokenizer is run over the result. Yields
    # (token number, token text) pairs, dropping position information.
    """
    Tokenize a Python source code string.

    Parameters
    ----------
    source : str
        The Python source code string.

    Returns
    -------
    tok_generator : Iterator[Tuple[int, str]]
        An iterator yielding all tokens with only toknum and tokval (Tuple[ing, str]).
    """
    # GH 59285
    # Escape characters, including backticks
    source = "".join(
        (
            create_valid_python_identifier(substring[1:-1])
            if is_backtick_quoted
            else substring
        )
        for is_backtick_quoted, substring in _split_by_backtick(source)
    )

    line_reader = StringIO(source).readline
    token_generator = tokenize.generate_tokens(line_reader)

    for toknum, tokval, _, _, _ in token_generator:
        yield toknum, tokval
