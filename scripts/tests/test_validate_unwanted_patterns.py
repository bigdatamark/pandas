# [MODULE DOC]
# Tests for the unwanted-pattern checks of the validation script: the misplaced-whitespace check on string
# concatenations, the no-default type check, and the warning-class check. Inputs are in-memory source snippets.
import io

import pytest

from scripts import validate_unwanted_patterns


class TestStringsWithWrongPlacedWhitespace:
    # [CLASS DOC]
    # Tests for the check on whitespace placement in concatenated strings: snippets that must produce no findings,
    # and snippets that must produce specific findings.
    @pytest.mark.parametrize(
        "data",
        [
            (
                r"""
    msg = (
        "foo\n"
        " bar"
    )
    """
            ),
            (
                """
    msg = (
        "foo"
        "  bar"
        "baz"
    )
    """
            ),
            (
                """
    msg = (
        f"foo"
        "  bar"
    )
    """
            ),
            (
                """
    msg = (
        "foo"
        f"  bar"
    )
    """
            ),
            (
                """
    msg = (
        "foo"
        rf"  bar"
    )
    """
            ),
        ],
    )
    def test_strings_with_wrong_placed_whitespace(self, data) -> None:
        # [FUNCTION DOC]
        # Verifies no findings for acceptable snippets: a newline escape followed by a leading space, two leading
        # spaces on the continuation (also with format and raw prefixes). Parametrized over the snippet.
        fd = io.StringIO(data.strip())
        result = list(
            validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd)
        )
        assert result == []

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                (
                    """
    msg = (
        "foo"
        " bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        f"foo"
        " bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        "foo"
        f" bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        f"foo"
        f" bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        "foo"
        rf" bar"
        " baz"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    ),
                    (
                        4,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    ),
                ],
            ),
            (
                (
                    """
    msg = (
        "foo"
        " bar"
        rf" baz"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    ),
                    (
                        4,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    ),
                ],
            ),
            (
                (
                    """
    msg = (
        "foo"
        rf" bar"
        rf" baz"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    ),
                    (
                        4,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    ),
                ],
            ),
        ],
    )
    def test_strings_with_wrong_placed_whitespace_raises(self, data, expected) -> None:
        # [FUNCTION DOC]
        # Verifies that snippets whose continuation string starts with a single space (with plain, format and raw
        # prefixes) yield findings with the expected line numbers (3, or 3 and 4) and message. Parametrized
        # over snippet and expected findings.
        fd = io.StringIO(data.strip())
        result = list(
            validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd)
        )
        assert result == expected


class TestNoDefaultUsedNotOnlyForTyping:
    # [CLASS DOC]
    # Tests for the check on the no-default sentinel type being used only for annotations.
    @pytest.mark.parametrize(
        "data",
        [
            (
                """
def f(
    a: int | NoDefault,
    b: float | lib.NoDefault = 0.1,
    c: pandas._libs.lib.NoDefault = lib.no_default,
) -> lib.NoDefault | None:
    pass
"""
            ),
            (
                """
# var = lib.NoDefault
# the above is incorrect
a: NoDefault | int
b: lib.NoDefault = lib.no_default
"""
            ),
        ],
    )
    def test_nodefault_used_not_only_for_typing(self, data) -> None:
        # [FUNCTION DOC]
        # Verifies no findings when the sentinel type appears only in parameter annotations, return annotations, variable
        # annotations or comments. Parametrized over the snippet.
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.nodefault_used_not_only_for_typing(fd))
        assert result == []

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                (
                    """
def f(
    a = lib.NoDefault,
    b: Any
        = pandas._libs.lib.NoDefault,
):
    pass
"""
                ),
                [
                    (2, "NoDefault is used not only for typing"),
                    (4, "NoDefault is used not only for typing"),
                ],
            ),
            (
                (
                    """
a: Any = lib.NoDefault
if a is NoDefault:
    pass
"""
                ),
                [
                    (1, "NoDefault is used not only for typing"),
                    (2, "NoDefault is used not only for typing"),
                ],
            ),
        ],
    )
    def test_nodefault_used_not_only_for_typing_raises(self, data, expected) -> None:
        # [FUNCTION DOC]
        # Verifies that uses outside annotations (a default value, an assignment value, a comparison) are reported with the
        # right line numbers and fixed message. Parametrized over snippet and expected findings.
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.nodefault_used_not_only_for_typing(fd))
        assert result == expected


@pytest.mark.parametrize("function", ["warnings.warn", "warn"])
@pytest.mark.parametrize("positional", [True, False])
@pytest.mark.parametrize(
    "category",
    [
        "FutureWarning",
        "DeprecationWarning",
        "PendingDeprecationWarning",
        "Pandas4Warning",
        "RuntimeWarning",
    ],
)
@pytest.mark.parametrize("pdlint_ignore", [True, False])
def test_doesnt_use_pandas_warnings(function, positional, category, pdlint_ignore):
    # [FUNCTION DOC]
    # Verifies the warning-class check over a grid of the call form (module attribute or imported name), positional or
    # keyword category, the category class (three built-in deprecation-type classes, one pandas-specific, one other),
    # and presence of the ignore marker comment. A finding is expected, once, at line one with a message starting with
    # "Don't use" plus the class name, only for the three built-in classes without the ignore marker; otherwise none.
    code = (
        f"{function}({'  # pdlint: ignore[warning_class]' if pdlint_ignore else ''}\n"
        f'   "message",\n'
        f"   {'' if positional else 'category='}{category},\n"
        f")\n"
    )
    flag_issue = (
        category in ["FutureWarning", "DeprecationWarning", "PendingDeprecationWarning"]
        and not pdlint_ignore
    )
    fd = io.StringIO(code)
    result = list(validate_unwanted_patterns.doesnt_use_pandas_warnings(fd))
    if flag_issue:
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1].startswith(f"Don't use {category}")
    else:
        assert len(result) == 0
