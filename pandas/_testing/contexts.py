# [MODULE DOC]
# HELPER: test support module with context managers and helpers that
# temporarily change the environment: decompress_file (open a compressed file),
# set_timezone (change the process time zone), with_csv_dialect (register a csv
# dialect) and raises_chained_assignment_error (build the right warning
# assertion for chained assignment). All are re-exported by the testing package.
# No import-time side effects beyond imports.
from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from typing import (
    IO,
    TYPE_CHECKING,
)

from pandas.compat import CHAINED_WARNING_DISABLED
from pandas.errors import ChainedAssignmentError

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas._typing import (
        BaseBuffer,
        CompressionOptions,
        FilePath,
    )


@contextmanager
def decompress_file(
    path: FilePath | BaseBuffer, compression: CompressionOptions
) -> Generator[IO[bytes]]:
    # [FUNCTION DOC]
    # HELPER: test helper (context manager, generator based). Parameters: path
    # (file path or binary buffer) and compression (a compression name such as
    # gzip, bz2, zip, xz, zstd, or none). Opens the path for binary reading
    # through the library's handle opener with the given decompression and yields
    # the underlying binary file object; the handle is closed on exit.
    # KNOWN ISSUE: the existing docstring has a Returns section, but the function
    # is a generator-based context manager that yields the file object.
    """
    Open a compressed file and return a file object.

    Parameters
    ----------
    path : str
        The path where the file is read from.

    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd', None}
        Name of the decompression to use

    Returns
    -------
    file object
    """
    with get_handle(path, "rb", compression=compression, is_text=False) as handle:
        yield handle.handle


@contextmanager
def set_timezone(tz: str) -> Generator[None]:
    # [FUNCTION DOC]
    # HELPER: test helper (context manager, generator based). Parameter tz is a
    # timezone name. Remembers the current value of the time zone environment
    # variable, sets the new one for the duration of the with block, and restores
    # the old one (or removes the variable if it was unset) afterwards, also on
    # error. Yields nothing. Process-global side effect on the environment and
    # on the C library's zone state; has no effect on platforms without the zone
    # reset function.
    """
    Context manager for temporarily setting a timezone.

    Parameters
    ----------
    tz : str
        A string representing a valid timezone.

    Examples
    --------
    >>> from datetime import datetime
    >>> from dateutil.tz import tzlocal
    >>> tzlocal().tzname(datetime(2021, 1, 1))  # doctest: +SKIP
    'IST'

    >>> with set_timezone("US/Eastern"):
    ...     tzlocal().tzname(datetime(2021, 1, 1))
    'EST'
    """
    import time

    def setTZ(tz) -> None:
        # [FUNCTION DOC]
        # HELPER: for the timezone context manager. Parameter tz is a zone name
        # or none. Does nothing on platforms lacking the time module's zone
        # reset function. If tz is none, removes the zone environment variable
        # (ignoring absence); otherwise sets it and, except on Windows, calls the
        # reset function so the new zone takes effect. Returns nothing.
        if hasattr(time, "tzset"):
            if tz is None:
                try:
                    del os.environ["TZ"]
                except KeyError:
                    pass
            else:
                os.environ["TZ"] = tz
                # Next line allows typing checks to pass on Windows
                if sys.platform != "win32":
                    time.tzset()

    orig_tz = os.environ.get("TZ")
    setTZ(tz)
    try:
        yield
    finally:
        setTZ(orig_tz)


@contextmanager
def with_csv_dialect(name: str, **kwargs) -> Generator[None]:
    # [FUNCTION DOC]
    # HELPER: test helper (context manager, generator based). Parameters: name
    # (dialect name) and keyword arguments describing the dialect (passed to the
    # standard library registration function). Raises a value error if the name
    # is one of the built-in dialect names (excel, excel-tab, unix). Registers the
    # dialect, yields nothing, and unregisters it on exit (also on error).
    """
    Context manager to temporarily register a CSV dialect for parsing CSV.

    Parameters
    ----------
    name : str
        The name of the dialect.
    kwargs : mapping
        The parameters for the dialect.

    Raises
    ------
    ValueError : the name of the dialect conflicts with a builtin one.

    See Also
    --------
    csv : Python's CSV library.
    """
    import csv

    _BUILTIN_DIALECTS = {"excel", "excel-tab", "unix"}

    if name in _BUILTIN_DIALECTS:
        raise ValueError("Cannot override builtin dialect.")

    csv.register_dialect(name, **kwargs)
    try:
        yield
    finally:
        csv.unregister_dialect(name)


def raises_chained_assignment_error(extra_warnings=(), extra_match=()):
    # [FUNCTION DOC]
    # HELPER: test helper. Parameters: extra_warnings (tuple of additional warning
    # classes expected, default empty) and extra_match (tuple of message
    # patterns for them, default empty). Returns a context manager. If the
    # compat flag says the chained-assignment warning is disabled: a do-nothing
    # context manager when there are no extra warnings, otherwise the warning
    # assertion for just the extra warnings with extra_match as the patterns.
    # Otherwise: the warning assertion for the chained-assignment warning class
    # (plus any extra classes) with the standard message pattern followed by the
    # extra patterns.
    # KNOWN ISSUE: when the warning is disabled and extra warnings are given,
    # extra_match is passed as the match even if empty, whereas a tuple length
    # must equal the expected-warning tuple length for the strict matching
    # mode; an empty default tuple combined with a tuple of warnings would
    # fail the length check.
    from pandas._testing import assert_produces_warning

    if CHAINED_WARNING_DISABLED:
        if not extra_warnings:
            from contextlib import nullcontext

            return nullcontext()
        else:
            return assert_produces_warning(
                extra_warnings,
                match=extra_match,
            )
    else:
        warning = ChainedAssignmentError
        match = (
            "A value is being set on a copy of a DataFrame or Series "
            "through chained assignment"
        )
        if extra_warnings:
            warning = (warning, *extra_warnings)  # type: ignore[assignment]
        return assert_produces_warning(
            warning,
            match=(match, *extra_match),
        )
