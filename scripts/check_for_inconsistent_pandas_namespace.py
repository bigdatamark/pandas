"""
Check that test suite file doesn't use the pandas namespace inconsistently.

We check for cases of ``Series`` and ``pd.Series`` appearing in the same file
(likewise for other pandas objects).

This is meant to be run as a pre-commit hook - to run it manually, you can do:

    pre-commit run inconsistent-namespace-usage --all-files

To automatically fixup a given file, you can pass `--replace`, e.g.

    python scripts/check_for_inconsistent_pandas_namespace.py test_me.py --replace

though note that you may need to manually fixup some imports and that you will also
need the additional dependency `tokenize-rt` (which is left out from the pre-commit
hook so that it uses the same virtualenv as the other local ones).

The general structure is similar to that of some plugins from
https://github.com/asottile/pyupgrade .
"""

# [MODULE DOC]
# Pre-commit script that detects test files using both a bare pandas object name
# (imported from pandas) and the same object through the pandas namespace alias in one
# file, and optionally rewrites the file to drop the namespace prefix. Public contents:
# the error message template, a record type for namespace attribute positions, an
# abstract-syntax-tree visitor collecting namespace usages and from-imports, a function
# doing the token-level rewrite, the checking function and the command line entry
# point. The rewrite needs the optional tokenize-rt package, imported lazily. No
# import-time side effects; runs the entry point when executed as a script.
import argparse
import ast
from collections.abc import (
    MutableMapping,
    Sequence,
)
import sys
from typing import NamedTuple

ERROR_MESSAGE = (
    "{path}:{lineno}:{col_offset}: Found both '{prefix}.{name}' and '{name}' in {path}"
)


class OffsetWithNamespace(NamedTuple):
    # [CLASS DOC]
    # HELPER: immutable record (line number, column offset, namespace identifier) locating one
    # attribute access on the pandas namespace, used as a dictionary key by the visitor.
    lineno: int
    col_offset: int
    namespace: str


class Visitor(ast.NodeVisitor):
    # [CLASS DOC]
    # HELPER: syntax tree visitor. State: pandas_namespace, mapping from position records
    # of attribute accesses on the names pandas or pd to the accessed attribute name; and
    # imported_from_pandas, set of all names imported by from-imports of any module whose
    # name contains the text pandas. Visits attributes and from-imports, always continuing
    # into child nodes.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: initializes the visitor with an empty position mapping and empty import set.
        self.pandas_namespace: MutableMapping[OffsetWithNamespace, str] = {}
        self.imported_from_pandas: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # [FUNCTION DOC]
        # HELPER: visitor hook for attribute access nodes. When the value is a plain name equal
        # to pandas or pd, records its position and namespace name against the attribute name.
        # Then visits children.
        if isinstance(node.value, ast.Name) and node.value.id in {"pandas", "pd"}:
            offset_with_namespace = OffsetWithNamespace(
                node.lineno, node.col_offset, node.value.id
            )
            self.pandas_namespace[offset_with_namespace] = node.attr
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # [FUNCTION DOC]
        # HELPER: visitor hook for from-import nodes. When the module name exists and contains the
        # text pandas (a substring match, so unrelated modules containing it also match), adds
        # all imported names to the set. Then visits children.
        if node.module is not None and "pandas" in node.module:
            self.imported_from_pandas.update(name.name for name in node.names)
        self.generic_visit(node)


def replace_inconsistent_pandas_namespace(visitor: Visitor, content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: rewrites source text to remove the namespace prefix on inconsistent usages.
    # Inputs: visitor, completed visitor; content, source text. Tokenizes the source (requires
    # the optional tokenize-rt package) and, scanning from the end, blanks the namespace token
    # and the following dot token for every recorded access whose attribute name was also
    # imported from pandas. Returns the new source text. Imports may need manual fixing.
    from tokenize_rt import (
        reversed_enumerate,
        src_to_tokens,
        tokens_to_src,
    )

    tokens = src_to_tokens(content)
    for n, i in reversed_enumerate(tokens):
        offset_with_namespace = OffsetWithNamespace(i.offset[0], i.offset[1], i.src)
        if (
            offset_with_namespace in visitor.pandas_namespace
            and visitor.pandas_namespace[offset_with_namespace]
            in visitor.imported_from_pandas
        ):
            # Replace `pd`
            tokens[n] = i._replace(src="")
            # Replace `.`
            tokens[n + 1] = tokens[n + 1]._replace(src="")

    new_src: str = tokens_to_src(tokens)
    return new_src


def check_for_inconsistent_pandas_namespace(
    content: str, path: str, *, replace: bool
) -> str | None:
    # [FUNCTION DOC]
    # Checks one file's source text. Inputs: content, source; path, file name for messages;
    # replace, required keyword flag. Parses and visits the source; returns none if no name
    # is both imported from pandas and used through the namespace. If inconsistent and replace
    # is false, writes one error message (for an arbitrary inconsistent name, without a trailing
    # newline) to standard output and exits the process with status one. If replace is true,
    # returns the rewritten source text.
    tree = ast.parse(content)

    visitor = Visitor()
    visitor.visit(tree)

    inconsistencies = visitor.imported_from_pandas.intersection(
        visitor.pandas_namespace.values()
    )

    if not inconsistencies:
        # No inconsistent namespace usage, nothing to replace.
        return None

    if not replace:
        inconsistency = inconsistencies.pop()
        lineno, col_offset, prefix = next(
            key for key, val in visitor.pandas_namespace.items() if val == inconsistency
        )
        msg = ERROR_MESSAGE.format(
            lineno=lineno,
            col_offset=col_offset,
            prefix=prefix,
            name=inconsistency,
            path=path,
        )
        sys.stdout.write(msg)
        sys.exit(1)

    return replace_inconsistent_pandas_namespace(visitor, content)


def main(argv: Sequence[str] | None = None) -> None:
    # [FUNCTION DOC]
    # Command line entry point. Input: argv, optional argument list (default: the process
    # arguments) with positional file paths and a replace flag. For each path, reads the file as
    # UTF-8, runs the check and, in replace mode when rewritten text is returned, writes it back to
    # the same file. In check mode the first inconsistent file terminates the process with status one.
    # Returns nothing.
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    for path in args.paths:
        with open(path, encoding="utf-8") as fd:
            content = fd.read()
        new_content = check_for_inconsistent_pandas_namespace(
            content, path, replace=args.replace
        )
        if not args.replace or new_content is None:
            continue
        with open(path, "w", encoding="utf-8") as fd:
            fd.write(new_content)


if __name__ == "__main__":
    main()
