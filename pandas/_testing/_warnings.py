# [MODULE DOC]
# HELPER: test support module providing context managers and checks for
# asserting which warnings code emits. Public contents: assert_produces_warning
# (the main context manager) and maybe_produces_warning (conditional wrapper),
# both re-exported by the testing package. Private helpers:
# _assert_caught_expected_warnings, _assert_caught_no_extra_warnings,
# _is_unexpected_warning and _assert_raised_with_correct_stacklevel. All
# failures are reported by raising assertion errors. No import-time side effects.
from __future__ import annotations

from contextlib import (
    AbstractContextManager,
    contextmanager,
    nullcontext,
)
import inspect
import re
import sys
from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    cast,
)
import warnings

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Sequence,
    )


@contextmanager
def assert_produces_warning(
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None = Warning,
    filter_level: Literal[
        "error", "ignore", "always", "default", "module", "once"
    ] = "always",
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: str | tuple[str | None, ...] | None = None,
    must_find_all_warnings: bool = True,
) -> Generator[list[warnings.WarningMessage]]:
    # [FUNCTION DOC]
    # HELPER: test helper (context manager, generator based). Records all
    # warnings raised inside the with block and, on exit, checks them.
    # Parameters: expected_warning (a warning class, a tuple of classes, or a
    # false value meaning no warning is expected; default is the base warning
    # class); filter_level (warning filter action applied inside the block,
    # default always); check_stacklevel (default true: each matching warning
    # must be attributed to the file of the calling test, otherwise an assertion
    # fails); raise_on_extra_warnings (default true: any warning that is not an
    # instance of the expected class fails the check); match (regular
    # expression text, or a tuple of them, that warning messages must match;
    # default none); must_find_all_warnings (default true: with a tuple of
    # expected classes each class must be seen, matched against the
    # corresponding entry of match, or the same pattern for all when match is
    # not a tuple; when false the tuple is treated as alternatives and a tuple
    # match is joined into one alternation pattern, ignoring empty entries).
    # Yields the list of recorded warning messages.
    # Raises an assertion error when an expected warning is missing, no message
    # matches, or unexpected warnings exist; the checks run in a final step, so
    # they also run when the body raised (and an assertion failure there can
    # replace the original exception).
    # Raises a value error if a tuple match has a different length than the
    # tuple of expected classes. Not thread safe, since it changes global
    # warning filters.
    # KNOWN ISSUE: the existing docstring describes the expected_warning as the
    # type of exception raised and refers to a nonexistent module, and its
    # trailing note uses an invalid directive spelling.
    """
    Context manager for running code expected to either raise a specific warning,
    multiple specific warnings, or not raise any warnings. Verifies that the code
    raises the expected warning(s), and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.

    Parameters
    ----------
    expected_warning : {Warning, False, tuple[Warning, ...], None}, default Warning
        The type of Exception raised. ``exception.Warning`` is the base
        class for all warnings. To raise multiple types of exceptions,
        pass them as a tuple. To check that no warning is returned,
        specify ``False`` or ``None``.
    filter_level : str or None, default "always"
        Specifies whether warnings are ignored, displayed, or turned
        into errors.
        Valid values are:

        * "error" - turns matching warnings into exceptions
        * "ignore" - discard the warning
        * "always" - always emit a warning
        * "default" - print the warning the first time it is generated
          from each location
        * "module" - print the warning the first time it is generated
          from each module
        * "once" - print the warning the first time it is generated

    check_stacklevel : bool, default True
        If True, displays the line that called the function containing
        the warning to show were the function is called. Otherwise, the
        line that implements the function is displayed.
    raise_on_extra_warnings : bool, default True
        Whether extra warnings not of the type `expected_warning` should
        cause the test to fail.
    match : {str, tuple[str, ...]}, optional
        Match warning message. If it's a tuple, it has to be the size of
        `expected_warning`. If additionally `must_find_all_warnings` is
        True, each expected warning's message gets matched with a respective
        match. Otherwise, multiple values get treated as an alternative.
    must_find_all_warnings : bool, default True
        If True and `expected_warning` is a tuple, each expected warning
        type must get encountered. Otherwise, even one expected warning
        results in success.

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    >>> with assert_produces_warning(False):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
    >>> with assert_produces_warning(UserWarning):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class 'UserWarning'.

    ..warn:: This is *not* thread-safe.
    """
    __tracebackhide__ = True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(filter_level)
        try:
            yield w
        finally:
            if expected_warning:
                if isinstance(expected_warning, tuple) and must_find_all_warnings:
                    match = (
                        match
                        if isinstance(match, tuple)
                        else (match,) * len(expected_warning)
                    )
                    for warning_type, warning_match in zip(
                        expected_warning, match, strict=True
                    ):
                        _assert_caught_expected_warnings(
                            caught_warnings=w,
                            expected_warning=warning_type,
                            match=warning_match,
                            check_stacklevel=check_stacklevel,
                        )
                else:
                    expected_warning = cast(
                        Union[type[Warning], tuple[type[Warning], ...]],
                        expected_warning,
                    )
                    match = (
                        "|".join(m for m in match if m)
                        if isinstance(match, tuple)
                        else match
                    )
                    _assert_caught_expected_warnings(
                        caught_warnings=w,
                        expected_warning=expected_warning,
                        match=match,
                        check_stacklevel=check_stacklevel,
                    )
            if raise_on_extra_warnings:
                _assert_caught_no_extra_warnings(
                    caught_warnings=w,
                    expected_warning=expected_warning,
                )


def maybe_produces_warning(
    warning: type[Warning], condition: bool, **kwargs
) -> AbstractContextManager:
    # [FUNCTION DOC]
    # HELPER: test helper. Parameters: warning (warning class expected),
    # condition (boolean) and extra keyword arguments passed through. Returns the
    # warning-assertion context manager configured with the class and keyword
    # arguments when condition is true; otherwise a do-nothing context manager.
    """
    Return a context manager that possibly checks a warning based on the condition
    """
    if condition:
        return assert_produces_warning(warning, **kwargs)
    else:
        return nullcontext()


def _assert_caught_expected_warnings(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning] | tuple[type[Warning], ...],
    match: str | None,
    check_stacklevel: bool,
) -> None:
    # [FUNCTION DOC]
    # HELPER: for assert_produces_warning. Keyword-only parameters:
    # caught_warnings (recorded warning messages), expected_warning (a class or
    # tuple of classes), match (regular expression text or none) and
    # check_stacklevel (bool). Scans the recorded warnings for instances of the
    # expected class(es); for each it optionally verifies the stack level and, if
    # a pattern is given, searches the message text with the pattern.
    # Raises an assertion error if no warning of the class was seen, or if a
    # non-empty pattern was given and no seen warning's message matched (the
    # error lists the messages that did not match). Returns nothing.
    """Assert that there was the expected warning among the caught warnings."""
    saw_warning = False
    matched_message = False
    unmatched_messages = []
    warning_name = (
        tuple(x.__name__ for x in expected_warning)
        if isinstance(expected_warning, tuple)
        else expected_warning.__name__
    )

    for actual_warning in caught_warnings:
        if issubclass(actual_warning.category, expected_warning):
            saw_warning = True

            if check_stacklevel:
                _assert_raised_with_correct_stacklevel(actual_warning)

            if match is not None:
                if re.search(match, str(actual_warning.message)):
                    matched_message = True
                else:
                    unmatched_messages.append(actual_warning.message)

    if not saw_warning:
        raise AssertionError(f"Did not see expected warning of class {warning_name!r}")

    if match and not matched_message:
        raise AssertionError(
            f"Did not see warning {warning_name!r} "
            f"matching '{match}'. The emitted warning messages are "
            f"{unmatched_messages}"
        )


def _assert_caught_no_extra_warnings(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: for assert_produces_warning. Keyword-only parameters:
    # caught_warnings (recorded warnings) and expected_warning (as given to the
    # context manager). Collects every warning judged unexpected by the private
    # predicate, except these deliberately ignored: resource warnings about
    # unclosed secure sockets (issue 44732), resource warnings whenever the
    # matplotlib module is loaded (it leaves font files open, issue 44844), and
    # encoding warnings (checked elsewhere in continuous integration). Raises an
    # assertion error listing the category name, message, file and line of the
    # remaining ones. Returns nothing.
    """Assert that no extra warnings apart from the expected ones are caught."""
    extra_warnings = []

    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            # GH#38630 pytest.filterwarnings does not suppress these.
            if actual_warning.category == ResourceWarning:
                # GH 44732: Don't make the CI flaky by filtering SSL-related
                # ResourceWarning from dependencies
                if "unclosed <ssl.SSLSocket" in str(actual_warning.message):
                    continue
                # GH 44844: Matplotlib leaves font files open during the entire process
                # upon import. Don't make CI flaky if ResourceWarning raised
                # due to these open files.
                if any("matplotlib" in mod for mod in sys.modules):
                    continue
            if actual_warning.category == EncodingWarning:
                # EncodingWarnings are checked in the CI
                # pyproject.toml errors on EncodingWarnings in pandas
                # Ignore EncodingWarnings from other libraries
                continue
            extra_warnings.append(
                (
                    actual_warning.category.__name__,
                    actual_warning.message,
                    actual_warning.filename,
                    actual_warning.lineno,
                )
            )

    if extra_warnings:
        raise AssertionError(f"Caused unexpected warning(s): {extra_warnings!r}")


def _is_unexpected_warning(
    actual_warning: warnings.WarningMessage,
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: for the extra-warning check. Parameters: actual_warning (a recorded
    # warning message) and expected_warning (class, tuple of classes, or a
    # false value). Returns true if no warning was expected at all (false value
    # given) or if the warning's category is not a subclass of the expected
    # class(es). Returns false otherwise.
    """Check if the actual warning issued is unexpected."""
    if actual_warning and not expected_warning:
        return True
    expected_warning = cast(type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning))


def _assert_raised_with_correct_stacklevel(
    actual_warning: warnings.WarningMessage,
) -> None:
    # [FUNCTION DOC]
    # HELPER: for the expected-warning check. Parameter actual_warning is a
    # recorded warning message. Finds the file of the code that called into the
    # context manager by going up exactly four frames from the current one
    # (a fixed depth that depends on the call chain inside the context manager
    # and the generator machinery), then asserts the warning's recorded file
    # name equals that caller's file, with a message naming both files and the
    # warning text. The frame reference is deleted afterwards to avoid cycles.
    # Raises an assertion error if the files differ. Returns nothing.
    # KNOWN ISSUE: the fixed frame depth makes the check fragile if the call
    # chain changes.
    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    for _ in range(4):
        frame = frame.f_back  # type: ignore[union-attr]
    try:
        caller_filename = inspect.getfile(frame)  # type: ignore[arg-type]
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    msg = (
        "Warning not set with correct stacklevel. "
        f"File where warning is raised: {actual_warning.filename} != "
        f"{caller_filename}. Warning message: {actual_warning.message}"
    )
    assert actual_warning.filename == caller_filename, msg
