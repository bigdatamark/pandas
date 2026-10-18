# [MODULE DOC]
# Exception and warning helpers used internally. Contents: a context manager
# that rewrites the first argument text of exceptions raised inside it, a function
# computing the stack level to report in warnings so they point at user code
# rather than library internals, and a context manager that rewrites matching
# warnings raised inside it. No import-time side effects.
from __future__ import annotations

import contextlib
import inspect
import os
import re
from typing import (
    TYPE_CHECKING,
    Any,
)
import warnings

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType


@contextlib.contextmanager
def rewrite_exception(old_name: str, new_name: str) -> Generator[None]:
    # [FUNCTION DOC]
    # Context manager rewriting exception text. Inputs: old_name and new_name,
    # strings; every occurrence of the old name in the first argument of any exception
    # raised in the managed block is replaced by the new name (the first argument is
    # converted to text first; remaining arguments are preserved) and the exception is
    # re-raised with the same type. Exceptions without arguments are re-raised
    # untouched. Only ordinary exceptions are intercepted. Yields nothing.
    """
    Rewrite the message of an exception.
    """
    try:
        yield
    except Exception as err:
        if not err.args:
            raise
        msg = str(err.args[0])
        msg = msg.replace(old_name, new_name)
        args: tuple[Any, ...] = (msg,)
        if len(err.args) > 1:
            args = args + err.args[1:]
        err.args = args
        raise


def find_stack_level() -> int:
    # [FUNCTION DOC]
    # Computes the number of stack frames, counting from this function's own frame,
    # that belong to the pandas package directory excluding its tests directory. The
    # walk stops at the first frame outside the package or inside the tests, so the
    # result is suitable as the stack level argument of a warning call so the
    # warning is attributed to the first caller outside pandas. Takes no input.
    # Imports the top-level package lazily to learn its directory. Deletes the frame
    # reference at the end to avoid reference cycles.
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """

    import pandas as pd

    pkg_dir = os.path.dirname(pd.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if filename.startswith(pkg_dir) and not filename.startswith(test_dir):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n


@contextlib.contextmanager
def rewrite_warning(
    target_message: str,
    target_category: type[Warning],
    new_message: str,
    new_category: type[Warning] | None = None,
) -> Generator[None]:
    # [FUNCTION DOC]
    # Context manager that replaces matching warnings raised inside its block.
    # Inputs: target_message, a regular expression searched in warning text;
    # target_category, warning class that must match exactly (not subclasses);
    # new_message, replacement text; new_category, replacement class, default none
    # meaning reuse target_category. All warnings raised in the block are captured
    # (and so not shown immediately); on exit, each is re-emitted with its original
    # file name and line number, replaced by the new message and category if it
    # matched, otherwise unchanged. Yields nothing.
    """
    Rewrite the message of a warning.

    Parameters
    ----------
    target_message : str
        Warning message to match.
    target_category : Warning
        Warning type to match.
    new_message : str
        New warning message to emit.
    new_category : Warning or None, default None
        New warning type to emit. When None, will be the same as target_category.
    """
    if new_category is None:
        new_category = target_category
    with warnings.catch_warnings(record=True) as record:
        yield
    if len(record) > 0:
        match = re.compile(target_message)
        for warning in record:
            if warning.category is target_category and re.search(
                match, str(warning.message)
            ):
                category = new_category
                message: Warning | str = new_message
            else:
                category, message = warning.category, warning.message
            warnings.warn_explicit(
                message=message,
                category=category,
                filename=warning.filename,
                lineno=warning.lineno,
            )
