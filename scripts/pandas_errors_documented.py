"""
Check that doc/source/reference/testing.rst documents
all exceptions and warnings in pandas/errors/__init__.py.

This is meant to be run as a pre-commit hook - to run it manually, you can do:

    pre-commit run pandas-errors-documented --all-files
"""

# [MODULE DOC]
# Pre-commit script checking that every exception and warning (every class defined, and every
# name imported by a from-import other than the future module) in the pandas errors module is
# mentioned in the testing reference documentation file. Public contents: a documentation path
# constant (resolved relative to the working directory at import), a function collecting
# defined names from source text, and a main function; run as a script it exits with status
# one after printing the undocumented names, else zero. No other import-time side effects.
from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

API_PATH = pathlib.Path("doc/source/reference/testing.rst").resolve()


def get_defined_errors(content: str) -> set[str]:
    # [FUNCTION DOC]
    # Collects names from source text: the names of all classes defined anywhere in the tree and
    # the names of all names brought in by from-imports (except from the future module). Input:
    # content, source text. Returns a set of strings.
    errors = set()
    for node in ast.walk(ast.parse(content)):
        if isinstance(node, ast.ClassDef):
            errors.add(node.name)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                errors.add(alias.name)
    return errors


def main(argv: Sequence[str] | None = None) -> None:
    # [FUNCTION DOC]
    # Entry point. Input: argv, optional argument list with one required positional path to the
    # errors module source. Computes the defined names from that file and the documented names
    # from the documentation file (the second dot-separated component of each line that contains
    # the word errors). Writes the set of undocumented names to standard output and exits with
    # status one if any are missing, otherwise exits with status zero. Never returns normally.
    # KNOWN ISSUE: despite the wording of the messages and docstring, imported helper names are
    # counted along with exception classes, so any imported name must also be documented.
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args(argv)
    with open(args.path, encoding="utf-8") as f:
        file_errors = get_defined_errors(f.read())
    with open(API_PATH, encoding="utf-8") as f:
        doc_errors = {
            line.split(".")[1].strip() for line in f.readlines() if "errors" in line
        }
    missing = file_errors.difference(doc_errors)
    if missing:
        sys.stdout.write(
            f"The following exceptions and/or warnings are not documented "
            f"in {API_PATH}: {missing}"
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
