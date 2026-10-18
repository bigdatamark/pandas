"""
Helpers for configuring locale settings.

Name `localization` is chosen to avoid overlap with builtin `locale` module.
"""

# [MODULE DOC]
# Helpers for working with operating system locales (used mainly by tests and
# formatting code). Public contents: set_locale (context manager that changes a
# locale temporarily), can_set_locale (probe), get_locales (list available
# locales) and the private helper _valid_locales. Relationships: reads the
# display encoding option from the config package to decode the system's
# locale list. Import-time side effects: none beyond imports.
from __future__ import annotations

from contextlib import contextmanager
import locale
import platform
import re
import subprocess
from typing import (
    TYPE_CHECKING,
    cast,
)

from pandas._config.config import options

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def set_locale(
    new_locale: str | tuple[str, str], lc_var: int = locale.LC_ALL
) -> Generator[str | tuple[str, str]]:
    # [FUNCTION DOC]
    # Context manager (generator based) that sets a locale for the duration of
    # the with block and restores the previous one afterwards, even if the block
    # raises. Parameters: new_locale (locale name string or a pair of language
    # and encoding) and lc_var (locale category integer, default all
    # categories). The previous locale is read with the set function (the get
    # function can disagree with it, issue 46595).
    # Yields the normalized locale as language code, a dot and the encoding when
    # the locale query returns both parts, otherwise the originally given value.
    # Raises the locale error (or a value error) from the standard library if
    # the locale cannot be set; in that case the restore step still runs.
    # Side effect: changes process-global locale state (not thread-safe).
    """
    Context manager for temporarily setting a locale.

    Parameters
    ----------
    new_locale : str or tuple
        A string of the form <language_country>.<encoding>. For example to set
        the current locale to US English with a UTF8 encoding, you would pass
        "en_US.UTF-8".
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Notes
    -----
    This is useful when you want to run a particular block of code under a
    particular locale, without globally setting the locale. This probably isn't
    thread-safe.
    """
    # getlocale is not always compliant with setlocale, use setlocale. GH#46595
    current_locale = locale.setlocale(lc_var)

    try:
        locale.setlocale(lc_var, new_locale)
        normalized_code, normalized_encoding = locale.getlocale()
        if normalized_code is not None and normalized_encoding is not None:
            yield f"{normalized_code}.{normalized_encoding}"
        else:
            yield new_locale
    finally:
        locale.setlocale(lc_var, current_locale)


def can_set_locale(lc: str, lc_var: int = locale.LC_ALL) -> bool:
    # [FUNCTION DOC]
    # Probes whether a locale can be set. Parameters: lc (locale name) and
    # lc_var (category, default all). Enters and immediately leaves the
    # temporary-locale context manager. Returns true if no error occurred, false
    # if a value error or locale error was raised. Side effect: briefly changes
    # and then restores the process locale.
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    bool
        Whether the passed locale can be set
    """
    try:
        with set_locale(lc, lc_var=lc_var):
            pass
    except (ValueError, locale.Error):
        # horrible name for an Exception subclass
        return False
    else:
        return True


def _valid_locales(locales: list[str] | str, normalize: bool) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: for get_locales. Filters candidate locale names down to those the
    # system accepts. Parameters: locales (an iterable of locale name strings;
    # the annotation also allows a single string but a string would be iterated
    # character by character) and normalize (if true each name is stripped and
    # then normalized by the standard library; otherwise only stripped).
    # Returns the list, preserving order, of names for which the probe function
    # succeeds. Side effect: temporarily sets each locale.
    # KNOWN ISSUE: the existing docstring says the input is a newline separated
    # string, but the code iterates over it as a sequence of names.
    """
    Return a list of normalized locales that do not throw an ``Exception``
    when set.

    Parameters
    ----------
    locales : str
        A string where each locale is separated by a newline.
    normalize : bool
        Whether to call ``locale.normalize`` on each locale.

    Returns
    -------
    valid_locales : list
        A list of valid locales.
    """
    return [
        loc
        for loc in (
            locale.normalize(loc.strip()) if normalize else loc.strip()
            for loc in locales
        )
        if can_set_locale(loc)
    ]


def get_locales(
    prefix: str | None = None,
    normalize: bool = True,
) -> list[str]:
    # [FUNCTION DOC]
    # Lists locales available on the machine that can actually be set.
    # Parameters: prefix (optional string; keep only locales matching the
    # prefix as a regular expression start, default none) and normalize (apply
    # the standard library's normalization to each name, default true).
    # On platforms other than Linux and macOS returns an empty list. Otherwise
    # runs the external locale-listing command, splits its byte output on
    # newlines and decodes each entry with the configured display encoding,
    # falling back to the windows-1252 codec for entries that fail to decode.
    # With no prefix, returns the valid ones among all entries. With a prefix,
    # joins the entries with newlines, extracts the matches of the pattern
    # prefix followed by any characters (per line), and returns the valid ones.
    # Raises the called-process error if the external command fails.
    # KNOWN ISSUE: the prefix is interpolated into a regular expression without
    # escaping.
    # KNOWN ISSUE: the existing docstring says that the normalize flag filters
    # to settable locales, but filtering by settability always happens.
    """
    Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return an empty list (no locale available, e.g. Windows)

    """
    if platform.system() in ("Linux", "Darwin"):
        raw_locales = subprocess.check_output(["locale", "-a"])
    else:
        # Other platforms e.g. windows platforms don't define "locale -a"
        #  Note: is_platform_windows causes circular import here
        return []

    try:
        # raw_locales is "\n" separated list of locales
        # it may contain non-decodable parts, so split
        # extract what we can and then rejoin.
        split_raw_locales = raw_locales.split(b"\n")
        out_locales = []
        for x in split_raw_locales:
            try:
                out_locales.append(str(x, encoding=cast(str, options.display.encoding)))
            except UnicodeError:
                # 'locale -a' is used to populated 'raw_locales' and on
                # Redhat 7 Linux (and maybe others) prints locale names
                # using windows-1252 encoding.  Bug only triggered by
                # a few special characters and when there is an
                # extensive list of installed locales.
                out_locales.append(str(x, encoding="windows-1252"))

    except TypeError:
        pass

    if prefix is None:
        return _valid_locales(out_locales, normalize)

    pattern = re.compile(f"{prefix}.*")
    found = pattern.findall("\n".join(out_locales))
    return _valid_locales(found, normalize)
