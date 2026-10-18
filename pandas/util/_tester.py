"""
Entrypoint for testing from the top-level namespace.
"""

# [MODULE DOC]
# Provides the pytest entry point exposed as the top-level pandas test
# function (the module attribute name is overridden to the top-level package
# by a decorator). Public content: the test function. Also defines a
# package-path constant, the directory containing the pandas package, used as
# the pytest target. Relies on the optional-dependency importer from the
# compat package. No import-time side effects.
from __future__ import annotations

import os
import sys

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import set_module

PKG = os.path.dirname(os.path.dirname(__file__))


@set_module("pandas")
def test(extra_args: list[str] | None = None, run_doctests: bool = False) -> None:  # noqa: PT028
    # [FUNCTION DOC]
    # Runs the pandas test suite with pytest and terminates the process.
    # Inputs: extra_args, an optional list of command line arguments (a single
    # non-list value is wrapped into a list); when given, it replaces the
    # default selection of "not slow, not network, not db" marks.
    # run_doctests, default false; when true, the command is replaced by one
    # that enables Python and Cython doctest collection and ignores the tests
    # directory.
    # KNOWN ISSUE: when run_doctests is true, any extra_args are discarded,
    # although the docstring says extra arguments may be combined with it.
    # Requires pytest and hypothesis (raises an import error through the
    # optional dependency importer if missing). Side effects: appends the
    # package path to the command (which mutates a caller-supplied list),
    # prints the command line, and exits the interpreter with pytest's exit
    # code. Never returns normally despite the declared None return type.
    """
    Run the pandas test suite using pytest.

    By default, runs with the marks -m "not slow and not network and not db"

    Parameters
    ----------
    extra_args : list[str], default None
        Extra marks to run the tests.
    run_doctests : bool, default False
        Whether to only run the Python and Cython doctests. If you would like to run
        both doctests/regular tests, just append "--doctest-modules"/"--doctest-cython"
        to extra_args.

    See Also
    --------
    pytest.main : The main entry point for pytest testing framework.

    Examples
    --------
    >>> pd.test()  # doctest: +SKIP
    running: pytest...
    """
    pytest = import_optional_dependency("pytest")
    import_optional_dependency("hypothesis")
    cmd = ["-m not slow and not network and not db"]
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args
    if run_doctests:
        cmd = [
            "--doctest-modules",
            "--doctest-cython",
            f"--ignore={os.path.join(PKG, 'tests')}",
        ]
    cmd += [PKG]
    joined = " ".join(cmd)
    print(f"running: pytest {joined}")
    sys.exit(pytest.main(cmd))


__all__ = ["test"]
