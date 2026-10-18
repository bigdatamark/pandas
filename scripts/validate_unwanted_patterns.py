#!/usr/bin/env python3
"""
Unwanted patterns test cases.

The reason this file exist despite the fact we already have
`ci/code_checks.sh`,
(see https://github.com/pandas-dev/pandas/blob/master/ci/code_checks.sh)

is that some of the test cases are more complex/impossible to validate via regex.
So this file is somewhat an extensions to `ci/code_checks.sh`
"""

# [MODULE DOC]
# Command line checker implementing source-code pattern checks that are too complex for regular
# expressions, complementing the shell-based code checks. Each check is a function taking an open text
# file and yielding pairs (line number, message). Public contents: a pattern matching deprecation-type warning
# class names; a set of private names exempt from the private-import check; a set of allowed private function
# calls; a helper measuring string literal prefixes; five check functions (private function use across
# modules, private import across modules, string concatenations with misplaced leading whitespace, misuse of the
# no-default sentinel type outside annotations, and use of built-in deprecation warnings instead of pandas ones); and
# main. As a script, it takes file paths, an output format template (default: path, line, message) and a required check name,
# runs that check over each file, prints violations and exits with status one if any were found. No import-time side effects.
import argparse
import ast
from collections.abc import (
    Callable,
    Iterable,
)
import re
import sys
import token
import tokenize
from typing import IO

DEPRECATION_WARNINGS_PATTERN = re.compile(
    r"(PendingDeprecation|Deprecation|Future)Warning"
)
PRIVATE_IMPORTS_TO_IGNORE: set[str] = {
    "_extension_array_shared_docs",
    "_index_shared_docs",
    "_interval_shared_docs",
    "_merge_doc",
    "_shared_docs",
    "_new_Index",
    "_new_PeriodIndex",
    "_pipe_template",
    "_apply_groupings_depr",
    "__main__",
    "_transform_template",
    "_get_plot_backend",
    "_matplotlib",
    "_arrow_utils",
    "_registry",
    "_test_parse_iso8601",
    "_testing",
    "_test_decorators",
    "__version__",  # check np.__version__ in compat.numpy.function
    "__git_version__",
    "_arrow_dtype_mapping",
    "_global_config",
    "_chained_assignment_msg",
    "_chained_assignment_method_msg",
    "_chained_assignment_method_update_msg",
    "_version_meson",
    # The numba extensions need this to mock the iloc object
    "_iLocIndexer",
    # TODO(4.0): GH#55043 - remove upon removal of CoW option
    "_get_option",
    "_fill_limit_area_1d",
    "_make_block",
    "_DatetimeTZBlock",
    "_check_pyarrow_available",
    "_parser",  # https://github.com/pandas-dev/pandas/issues/60833
    "_safe_fill_null",
}


def _get_literal_string_prefix_len(token_string: str) -> int:
    # [FUNCTION DOC]
    # HELPER: returns the number of characters before the first quote character of a string token text, i.e. the length of
    # the literal prefix such as raw or format markers. Input: token_string. Returns zero if the text contains no quote.
    """
    Getting the length of the literal string prefix.

    Parameters
    ----------
    token_string : str
        String to check.

    Returns
    -------
    int
        Length of the literal string prefix.

    Examples
    --------
    >>> example_string = "'Hello world'"
    >>> _get_literal_string_prefix_len(example_string)
    0
    >>> example_string = "r'Hello world'"
    >>> _get_literal_string_prefix_len(example_string)
    1
    """
    try:
        return min(
            token_string.find(quote)
            for quote in (r"'", r'"')
            if token_string.find(quote) >= 0
        )
    except ValueError:
        return 0


PRIVATE_FUNCTIONS_ALLOWED = {"sys._getframe"}  # no known alternative


def private_function_across_module(file_obj: IO[str]) -> Iterable[tuple[int, str]]:
    # [FUNCTION DOC]
    # Check: private function called through an imported module. Input: file_obj, readable text file. Parses the whole
    # file; collects imported module names (the alias if given, otherwise the full name; the imported names of from-imports
    # also count); for each call of the form name.attribute, ignores names starting with an uppercase letter (class
    # case), dunder attributes and the allowed private calls, and yields the call's line number and a message naming the
    # function when the name was imported and the attribute starts with an underscore. Raises a syntax error for
    # unparseable files.
    """
    Checking that a private function is not used across modules.
    Parameters
    ----------
    file_obj : IO
        File-like object containing the Python code to validate.
    Yields
    ------
    line_number : int
        Line number of the private function that is used across modules.
    msg : str
        Explanation of the error.
    """
    contents = file_obj.read()
    tree = ast.parse(contents)

    imported_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for module in node.names:
                module_fqdn = module.name if module.asname is None else module.asname
                imported_modules.add(module_fqdn)

        if not isinstance(node, ast.Call):
            continue

        try:
            module_name = node.func.value.id
            function_name = node.func.attr
        except AttributeError:
            continue

        # Exception section #

        # (Debatable) Class case
        if module_name[0].isupper():
            continue
        # (Debatable) Dunder methods case
        elif function_name.startswith("__") and function_name.endswith("__"):
            continue
        elif module_name + "." + function_name in PRIVATE_FUNCTIONS_ALLOWED:
            continue

        if module_name in imported_modules and function_name.startswith("_"):
            yield (node.lineno, f"Private function '{module_name}.{function_name}'")


def private_import_across_module(file_obj: IO[str]) -> Iterable[tuple[int, str]]:
    # [FUNCTION DOC]
    # Check: import of a private name. Input: file_obj, readable text file. Parses the file; for each import statement
    # and each imported name, takes the last dot-separated component; skips names in the exemption set; yields the
    # statement's line number and a message naming the internal function when the component starts with an underscore.
    """
    Checking that a private function is not imported across modules.
    Parameters
    ----------
    file_obj : IO
        File-like object containing the Python code to validate.
    Yields
    ------
    line_number : int
        Line number of import statement, that imports the private function.
    msg : str
        Explanation of the error.
    """
    contents = file_obj.read()
    tree = ast.parse(contents)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue

        for module in node.names:
            module_name = module.name.split(".")[-1]
            if module_name in PRIVATE_IMPORTS_TO_IGNORE:
                continue

            if module_name.startswith("_"):
                yield (node.lineno, f"Import of internal function {module_name!r}")


def strings_with_wrong_placed_whitespace(
    file_obj: IO[str],
) -> Iterable[tuple[int, str]]:
    # [FUNCTION DOC]
    # Check: implicitly concatenated string literals where the following piece starts with a space rather than the
    # previous piece ending with one. Input: file_obj, readable text file. Tokenizes the file and looks at every
    # triple of consecutive tokens made of a string, a non-logical newline and a string. Quotes and prefix are
    # stripped from both strings, and the pair is flagged when the first piece does not end with a newline escape
    # or a space, neither piece has two leading or trailing spaces, and the second piece begins with a single space.
    # Yields the second string's line number and a fixed message.
    # KNOWN ISSUE: the docstring's illustrative bad and good examples are textually identical, so they do not show
    # the difference they describe.
    """
    Test case for leading spaces in concated strings.

    For example:

    >>> rule = "We want the space at the end of the line, not at the beginning"

    Instead of:

    >>> rule = "We want the space at the end of the line, not at the beginning"

    Parameters
    ----------
    file_obj : IO
        File-like object containing the Python code to validate.

    Yields
    ------
    line_number : int
        Line number of unconcatenated string.
    msg : str
        Explanation of the error.
    """

    def has_wrong_whitespace(first_line: str, second_line: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: decides whether a pair of adjacent string pieces violates the whitespace convention. Inputs:
        # first_line, second_line, the unquoted text of the pieces. Returns false if the first ends with a newline escape,
        # or either piece starts or ends with two spaces; true if the first does not end with a space and the second
        # starts with one; false otherwise.
        """
        Checking if the two lines are mattching the unwanted pattern.

        Parameters
        ----------
        first_line : str
            First line to check.
        second_line : str
            Second line to check.

        Returns
        -------
        bool
            True if the two received string match, an unwanted pattern.

        Notes
        -----
        The unwanted pattern that we are trying to catch is if the spaces in
        a string that is concatenated over multiple lines are placed at the
        end of each string, unless this string is ending with a
        newline character (\n).

        For example, this is bad:

        >>> rule = "We want the space at the end of the line, not at the beginning"

        And what we want is:

        >>> rule = "We want the space at the end of the line, not at the beginning"

        And if the string is ending with a new line character (\n) we
        do not want any trailing whitespaces after it.

        For example, this is bad:

        >>> rule = (
        ...     "We want the space at the begging of "
        ...     "the line if the previous line is ending with a \n "
        ...     "not at the end, like always"
        ... )

        And what we do want is:

        >>> rule = (
        ...     "We want the space at the begging of "
        ...     "the line if the previous line is ending with a \n"
        ...     " not at the end, like always"
        ... )
        """
        if first_line.endswith(r"\n"):
            return False
        elif first_line.startswith("  ") or second_line.startswith("  "):
            return False
        elif first_line.endswith("  ") or second_line.endswith("  "):
            return False
        elif (not first_line.endswith(" ")) and second_line.startswith(" "):
            return True
        return False

    tokens: list = list(tokenize.generate_tokens(file_obj.readline))

    for first_token, second_token, third_token in zip(
        tokens, tokens[1:], tokens[2:], strict=False
    ):
        # Checking if we are in a block of concated string
        if (
            first_token.type == third_token.type == token.STRING
            and second_token.type == token.NL
        ):
            # Striping the quotes, with the string literal prefix
            first_string: str = first_token.string[
                _get_literal_string_prefix_len(first_token.string) + 1 : -1
            ]
            second_string: str = third_token.string[
                _get_literal_string_prefix_len(third_token.string) + 1 : -1
            ]

            if has_wrong_whitespace(first_string, second_string):
                yield (
                    third_token.start[0],
                    (
                        "String has a space at the beginning instead "
                        "of the end of the previous string."
                    ),
                )


def nodefault_used_not_only_for_typing(file_obj: IO[str]) -> Iterable[tuple[int, str]]:
    # [FUNCTION DOC]
    # Check: the no-default sentinel type name used outside type annotations. Input: file_obj, readable text file. Parses the file
    # and walks the syntax tree iteratively with a flag for whether the current node lies within a variable annotation or return
    # annotation. Any plain name or attribute access named NoDefault outside annotations yields its line number and a fixed message.
    """Test case where pandas._libs.lib.NoDefault is not used for typing.

    Parameters
    ----------
    file_obj : IO
        File-like object containing the Python code to validate.

    Yields
    ------
    line_number : int
        Line number of misused lib.NoDefault.
    msg : str
        Explanation of the error.
    """
    contents = file_obj.read()
    tree = ast.parse(contents)
    in_annotation = False
    nodes: list[tuple[bool, ast.AST]] = [(in_annotation, tree)]

    while nodes:
        in_annotation, node = nodes.pop()
        if not in_annotation and (
            (
                isinstance(node, ast.Name)  # Case `NoDefault`
                and node.id == "NoDefault"
            )
            or (
                isinstance(node, ast.Attribute)  # Cases e.g. `lib.NoDefault`
                and node.attr == "NoDefault"
            )
        ):
            yield (node.lineno, "NoDefault is used not only for typing")

        # This part is adapted from
        # https://github.com/asottile/pyupgrade/blob/5495a248f2165941c5d3b82ac3226ba7ad1fa59d/pyupgrade/_data.py#L70-L113
        for name in reversed(node._fields):
            value = getattr(node, name)
            if name in {"annotation", "returns"}:
                next_in_annotation = True
            else:
                next_in_annotation = in_annotation
            if isinstance(value, ast.AST):
                nodes.append((next_in_annotation, value))
            elif isinstance(value, list):
                nodes.extend(
                    (next_in_annotation, value)
                    for value in reversed(value)
                    if isinstance(value, ast.AST)
                )


def doesnt_use_pandas_warnings(file_obj: IO[str]) -> Iterable[tuple[int, str]]:
    # [FUNCTION DOC]
    # Check: use of built-in deprecation or future warning classes. Input: file_obj, readable text file. Parses the file and
    # inspects calls to warn (either through the warnings module or imported directly). Calls whose source lines, from the call start
    # through the line after its end, contain an ignore marker comment are skipped. For other calls, collects every positional
    # argument that is a plain name, and the value of the category keyword if it is a plain name, and yields (line number,
    # message) for each whose name starts with a pending-deprecation, deprecation or future warning name, telling the user to use a pandas-specific warning.
    # KNOWN ISSUE: calls whose function is an attribute of something other than a plain name, or any other expression kind, are not
    # skipped by the filtering, so they are examined as if they were warn calls.
    """
    Checking that pandas-specific warnings are used for deprecations.

    Parameters
    ----------
    file_obj : IO
        File-like object containing the Python code to validate.

    Yields
    ------
    line_number : int
        Line number of the warning.
    msg : str
        Explanation of the error.
    """
    contents = file_obj.read()
    lines = contents.split("\n")
    tree = ast.parse(contents)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            # Check for `warnings.warn`.
            if node.func.value.id != "warnings" or node.func.attr != "warn":
                continue
        elif isinstance(node.func, ast.Name):
            # Check for just `warn` when using `from warnings import warn`.
            if node.func.id != "warn":
                continue
        if any(
            "# pdlint: ignore[warning_class]" in lines[k]
            for k in range(node.lineno - 1, node.end_lineno + 1)
        ):
            continue
        values = [arg.id for arg in node.args if isinstance(arg, ast.Name)] + [
            kw.value.id for kw in node.keywords if kw.arg == "category"
        ]
        for value in values:
            matches = re.match(DEPRECATION_WARNINGS_PATTERN, value)
            if matches is not None:
                yield (
                    node.lineno,
                    f"Don't use {matches[0]}, use a pandas-specific warning in "
                    f"pd.errors instead. You can add "
                    f"`# pdlint: ignore[warning_class]` to override.",
                )


def main(
    function: Callable[[IO[str]], Iterable[tuple[int, str]]],
    source_path: str,
    output_format: str,
) -> bool:
    # [FUNCTION DOC]
    # Runs one check over a list of files and prints violations. Inputs: function, the check to run; source_path, the list of
    # file paths (despite the docstring calling it a single path); output_format, a template string with fields for the path,
    # line number and message. Each file is read as UTF-8; each violation is printed with the template. Returns true if any
    # violation was found.
    # KNOWN ISSUE: the docstring lists two parameters (file extensions and excluded paths) that the function does not
    # accept, describes the source path as a string naming a file or directory, and documents a value error that is never raised.
    """
    Main entry point of the script.

    Parameters
    ----------
    function : Callable
        Function to execute for the specified validation type.
    source_path : str
        Source path representing path to a file/directory.
    output_format : str
        Output format of the error message.
    file_extensions_to_check : str
        Comma separated values of what file extensions to check.
    excluded_file_paths : str
        Comma separated values of what file paths to exclude during the check.

    Returns
    -------
    bool
        True if found any patterns are found related to the given function.

    Raises
    ------
    ValueError
        If the `source_path` is not pointing to existing file/directory.
    """
    is_failed: bool = False

    for file_path in source_path:
        with open(file_path, encoding="utf-8") as file_obj:
            for line_number, msg in function(file_obj):
                is_failed = True
                print(
                    output_format.format(
                        source_path=file_path, line_number=line_number, msg=msg
                    )
                )

    return is_failed


if __name__ == "__main__":
    available_validation_types: list[str] = [
        "private_function_across_module",
        "private_import_across_module",
        "strings_with_wrong_placed_whitespace",
        "nodefault_used_not_only_for_typing",
        "doesnt_use_pandas_warnings",
    ]

    parser = argparse.ArgumentParser(description="Unwanted patterns checker.")

    parser.add_argument("paths", nargs="*", help="Source paths of files to check.")
    parser.add_argument(
        "--format",
        "-f",
        default="{source_path}:{line_number}: {msg}",
        help="Output format of the error message.",
    )
    parser.add_argument(
        "--validation-type",
        "-vt",
        choices=available_validation_types,
        required=True,
        help="Validation test case to check.",
    )

    args = parser.parse_args()

    sys.exit(
        main(
            function=globals().get(args.validation_type),
            source_path=args.paths,
            output_format=args.format,
        )
    )
