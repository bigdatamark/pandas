"""
Check that test names start with `test`, and that test classes start with `Test`.

This is meant to be run as a pre-commit hook - to run it manually, you can do:

    pre-commit run check-test-naming --all-files

NOTE: if this finds a false positive, you can add the comment `# not a test` to the
class or function definition. Though hopefully that shouldn't be necessary.
"""

# [MODULE DOC]
# Pre-commit script verifying that, in test files, top-level functions start with
# the test prefix and top-level classes start with the capitalized test prefix, and
# that the methods of non-test classes do too unless they are used by name elsewhere.
# Objects can be exempted by a trailing pragma comment on the definition line. Public
# contents: the pragma constant, name-collection and decorator-detection helpers,
# two predicates for misnamed functions and classes, a per-file checker, and a command
# line block run when executed as a script (it only looks at files whose base name starts
# with test and ends with the Python extension). Exit status is non-zero if any problem
# was printed. No import-time side effects.
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

PRAGMA = "# not a test"


def _find_names(node: ast.Module) -> Iterator[str]:
    # [FUNCTION DOC]
    # HELPER: yields every identifier used in a syntax tree: names of plain-name nodes and
    # attribute names, from a full walk of the tree.
    for _node in ast.walk(node):
        if isinstance(_node, ast.Name):
            yield _node.id
        elif isinstance(_node, ast.Attribute):
            yield _node.attr


def _is_fixture(node: ast.expr) -> bool:
    # [FUNCTION DOC]
    # HELPER: tells whether a decorator expression denotes a pytest fixture, either bare or as
    # a call with arguments, by checking for an attribute named fixture on the name pytest.
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "fixture"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
    )


def _is_register_dtype(node):
    # [FUNCTION DOC]
    # HELPER: tells whether a decorator expression is the plain name used to register an
    # extension dtype class. Returns a bool.
    return isinstance(node, ast.Name) and node.id == "register_extension_dtype"


def is_misnamed_test_func(
    node: ast.expr | ast.stmt, names: Sequence[str], line: str
) -> bool:
    # [FUNCTION DOC]
    # Tells whether a syntax node is a plain function definition that should have been named
    # as a test but is not. Inputs: node; names, all identifiers seen in the module; line,
    # the source text of the definition line. True when the node is a function whose name does
    # not start with the test prefix, does not appear among the identifiers, is not a fixture, has
    # no pragma comment on its line, and is not a setup or teardown method name (method or class
    # variants).
    return (
        isinstance(node, ast.FunctionDef)
        and not node.name.startswith("test")
        and names.count(node.name) == 0
        and not any(_is_fixture(decorator) for decorator in node.decorator_list)
        and PRAGMA not in line
        and node.name
        not in ("teardown_method", "setup_method", "teardown_class", "setup_class")
    )


def is_misnamed_test_class(
    node: ast.expr | ast.stmt, names: Sequence[str], line: str
) -> bool:
    # [FUNCTION DOC]
    # Tells whether a syntax node is a class definition that should have been named as a test
    # class but is not. Inputs as for the function version. True when the class name does not
    # start with the capitalized test prefix, does not appear among identifiers, is not decorated
    # with the dtype registration decorator, and has no pragma comment on its line.
    return (
        isinstance(node, ast.ClassDef)
        and not node.name.startswith("Test")
        and names.count(node.name) == 0
        and not any(_is_register_dtype(decorator) for decorator in node.decorator_list)
        and PRAGMA not in line
    )


def main(content: str, file: str) -> int:
    # [FUNCTION DOC]
    # Checks one module's source. Inputs: content, source text; file, path for messages. Returns
    # zero when clean, otherwise one. For each top-level node, prints a message with file, line
    # and column for a misnamed function or class. For top-level classes that are themselves
    # eligible (not used by name, not registration-decorated, no pragma), also checks each
    # direct method: a misnamed method is reported unless some file under the pandas tests
    # directory (searched relative to the current working directory) contains a self-attribute
    # access with that method name, since a parent class may call it. Side effects: reads
    # many files, prints to standard output.
    lines = content.splitlines()
    tree = ast.parse(content)
    names = list(_find_names(tree))
    ret = 0
    for node in tree.body:
        if is_misnamed_test_func(node, names, lines[node.lineno - 1]):
            print(
                f"{file}:{node.lineno}:{node.col_offset} "
                "found test function which does not start with 'test'"
            )
            ret = 1
        elif is_misnamed_test_class(node, names, lines[node.lineno - 1]):
            print(
                f"{file}:{node.lineno}:{node.col_offset} "
                "found test class which does not start with 'Test'"
            )
            ret = 1
        if (
            isinstance(node, ast.ClassDef)
            and names.count(node.name) == 0
            and not any(
                _is_register_dtype(decorator) for decorator in node.decorator_list
            )
            and PRAGMA not in lines[node.lineno - 1]
        ):
            for _node in node.body:
                if is_misnamed_test_func(_node, names, lines[_node.lineno - 1]):
                    # It could be that this function is used somewhere by the
                    # parent class. For example, there might be a base class
                    # with
                    #
                    # class Foo:
                    #     def foo(self):
                    #          assert 1+1==2
                    #     def test_foo(self):
                    #         self.foo()
                    #
                    # and then some subclass overwrites `foo`. So, we check that
                    # `self.foo` doesn't appear in any of the test classes.
                    # Note some false negatives might get through, but that's OK.
                    # This is good enough that has helped identify several examples
                    # of tests not being run.
                    assert isinstance(_node, ast.FunctionDef)  # help mypy
                    should_continue = False
                    for _file in (Path("pandas") / "tests").rglob("*.py"):
                        with open(os.path.join(_file), encoding="utf-8") as fd:
                            _content = fd.read()
                        if f"self.{_node.name}" in _content:
                            should_continue = True
                            break
                    if should_continue:
                        continue

                    print(
                        f"{file}:{_node.lineno}:{_node.col_offset} "
                        "found test function which does not start with 'test'"
                    )
                    ret = 1
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    ret = 0

    for file in args.paths:
        filename = os.path.basename(file)
        if not (filename.startswith("test") and filename.endswith(".py")):
            continue
        with open(file, encoding="utf-8") as fd:
            content = fd.read()
        ret |= main(content, file)

    sys.exit(ret)
