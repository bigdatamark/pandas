""" "
Test module for testing ``pandas._testing.assert_produces_warning``.
"""

# [MODULE DOC]
# Tests for the testing-utility context manager that asserts a block of code produces certain warnings: category and message matching, failure messages for unmatched, missing and extra warnings, handling of multiple expected categories (all must be found unless relaxed), expecting no warnings (false or none), and behavior when the block raises an exception. Defines a fixture of pairs of different warning categories and a helper function that emits two warnings.

import warnings

import pytest

from pandas.errors import (
    DtypeWarning,
    PerformanceWarning,
)

import pandas._testing as tm


@pytest.fixture(
    params=[
        (RuntimeWarning, UserWarning),
        (UserWarning, FutureWarning),
        (FutureWarning, RuntimeWarning),
        (DeprecationWarning, PerformanceWarning),
        (PerformanceWarning, FutureWarning),
        (DtypeWarning, DeprecationWarning),
        (ResourceWarning, DeprecationWarning),
        (FutureWarning, DeprecationWarning),
    ],
    ids=lambda x: type(x).__name__,
)
def pair_different_warnings(request):
    # [FUNCTION DOC]
    # HELPER: fixture returning a pair of different warning categories (expected, other) from a parametrized list that includes runtime, user, future, deprecation, performance, dtype and resource warnings in various pairings. Used to test how different warnings are handled by the assertion context manager.
    # KNOWN ISSUE: the identifier function for the parametrization applies a type-name function to the whole parameter tuple, so every test id is derived from the tuple type name rather than from the warning classes, and the docstring has a typo ("pair or different").
    """
    Return pair or different warnings.

    Useful for testing how several different warnings are handled
    in tm.assert_produces_warning.
    """
    return request.param


def f():
    # [FUNCTION DOC]
    # HELPER: emits two warnings, a future warning with message f1 and a runtime warning with message f2. Takes no parameters and returns nothing; side effect: warnings issued.
    warnings.warn("f1", FutureWarning)  # pdlint: ignore[warning_class]
    warnings.warn("f2", RuntimeWarning)


def test_assert_produces_warning_honors_filter():
    # [FUNCTION DOC]
    # Expecting only the runtime warning while the helper also emits a future warning raises an assertion error saying unexpected warnings were caused; with the extra-warning check disabled it passes.
    # Raise by default.
    msg = r"Caused unexpected warning\(s\)"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            f()

    with tm.assert_produces_warning(RuntimeWarning, raise_on_extra_warnings=False):
        f()


@pytest.mark.parametrize(
    "category",
    [
        RuntimeWarning,
        ResourceWarning,
        UserWarning,
        FutureWarning,
        DeprecationWarning,
        PerformanceWarning,
        DtypeWarning,
    ],
)
@pytest.mark.parametrize(
    "message, match",
    [
        ("", None),
        ("", ""),
        ("Warning message", r".*"),
        ("Warning message", "War"),
        ("Warning message", r"[Ww]arning"),
        ("Warning message", "age"),
        ("Warning message", r"age$"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"^Mes.*\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}\s\S+"),
        ("Message, which we do not match", None),
    ],
)
def test_catch_warning_category_and_match(category, message, match):
    # [FUNCTION DOC]
    # Parametrized over seven warning categories and eleven (message, pattern) pairs, including no pattern, empty pattern and regular expressions. Emitting the message with that category inside the context manager expecting the category and pattern passes.
    with tm.assert_produces_warning(category, match=match):
        warnings.warn(message, category)


def test_fail_to_match_runtime_warning():
    # [FUNCTION DOC]
    # Expecting a runtime warning with a non-matching pattern when two non-matching runtime warnings were emitted raises an assertion error saying the expected warning was not seen and listing the emitted messages.
    category = RuntimeWarning
    match = "Did not see this warning"
    unmatched = (
        r"Did not see warning 'RuntimeWarning' matching 'Did not see this warning'. "
        r"The emitted warning messages are "
        r"\[RuntimeWarning\('This is not a match.'\), "
        r"RuntimeWarning\('Another unmatched warning.'\)\]"
    )
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn("This is not a match.", category)
            warnings.warn("Another unmatched warning.", category)


def test_fail_to_match_future_warning():
    # [FUNCTION DOC]
    # Same as above for a future warning and a short pattern.
    category = FutureWarning
    match = "Warning"
    unmatched = (
        r"Did not see warning 'FutureWarning' matching 'Warning'. "
        r"The emitted warning messages are "
        r"\[FutureWarning\('This is not a match.'\), "
        r"FutureWarning\('Another unmatched warning.'\)\]"
    )
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn("This is not a match.", category)
            warnings.warn("Another unmatched warning.", category)


def test_fail_to_match_resource_warning():
    # [FUNCTION DOC]
    # Same as above for a resource warning and a digits pattern (the escaped pattern is shown in the message).
    category = ResourceWarning
    match = r"\d+"
    unmatched = (
        r"Did not see warning 'ResourceWarning' matching '\\d\+'. "
        r"The emitted warning messages are "
        r"\[ResourceWarning\('This is not a match.'\), "
        r"ResourceWarning\('Another unmatched warning.'\)\]"
    )
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn("This is not a match.", category)
            warnings.warn("Another unmatched warning.", category)


def test_fail_to_catch_actual_warning(pair_different_warnings):
    # [FUNCTION DOC]
    # Parametrized over pairs. Expecting one category but emitting another raises an assertion error saying the expected warning class was not seen.
    expected_category, actual_category = pair_different_warnings
    match = "Did not see expected warning of class"
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn("warning message", actual_category)


def test_ignore_extra_warning(pair_different_warnings):
    # [FUNCTION DOC]
    # Parametrized over pairs. Emitting the expected plus an extra warning passes when the extra-warning check is disabled.
    expected_category, extra_category = pair_different_warnings
    with tm.assert_produces_warning(expected_category, raise_on_extra_warnings=False):
        warnings.warn("Expected warning", expected_category)
        warnings.warn("Unexpected warning OK", extra_category)


def test_raise_on_extra_warning(pair_different_warnings):
    # [FUNCTION DOC]
    # Parametrized over pairs. Emitting the expected plus an extra warning raises an assertion error about unexpected warnings by default.
    expected_category, extra_category = pair_different_warnings
    match = r"Caused unexpected warning\(s\)"
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn("Expected warning", expected_category)
            warnings.warn("Unexpected warning NOT OK", extra_category)


def test_same_category_different_messages_first_match():
    # [FUNCTION DOC]
    # Three warnings of the same category where only the first matches the pattern: passes.
    category = UserWarning
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Match this", category)
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)


def test_same_category_different_messages_last_match():
    # [FUNCTION DOC]
    # Three warnings of the same category where only the last matches the pattern: passes.
    category = DeprecationWarning
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)
        warnings.warn("Match this", category)


def test_match_multiple_warnings():
    # [FUNCTION DOC]
    # Issue 47829. Two expected categories with a shared pattern: both emitted warnings match, so it passes.
    # https://github.com/pandas-dev/pandas/issues/47829
    category = (FutureWarning, UserWarning)
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Match this", FutureWarning)  # pdlint: ignore[warning_class]
        warnings.warn("Match this too", UserWarning)


def test_must_match_multiple_warnings():
    # [FUNCTION DOC]
    # Issue 56555. With two expected categories, emitting only a matching future warning raises an assertion error saying the user warning class was not seen.
    # https://github.com/pandas-dev/pandas/issues/56555
    category = (FutureWarning, UserWarning)
    msg = "Did not see expected warning of class 'UserWarning'"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(category, match=r"^Match this"):
            warnings.warn("Match this", FutureWarning)  # pdlint: ignore[warning_class]


def test_must_match_multiple_warnings_messages():
    # [FUNCTION DOC]
    # Issue 56555. When a matching future warning and a non-matching user warning are emitted, the assertion error lists the emitted messages of the unmatched user warning.
    # https://github.com/pandas-dev/pandas/issues/56555
    category = (FutureWarning, UserWarning)
    msg = r"The emitted warning messages are \[UserWarning\('Not this'\)\]"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(category, match=r"^Match this"):
            warnings.warn("Match this", FutureWarning)  # pdlint: ignore[warning_class]
            warnings.warn("Not this", UserWarning)


def test_allow_partial_match_for_multiple_warnings():
    # [FUNCTION DOC]
    # Issue 56555. With the must-find-all option disabled, emitting only one of two expected categories passes.
    # https://github.com/pandas-dev/pandas/issues/56555
    category = (FutureWarning, UserWarning)
    with tm.assert_produces_warning(
        category, match=r"^Match this", must_find_all_warnings=False
    ):
        warnings.warn("Match this", FutureWarning)  # pdlint: ignore[warning_class]


def test_allow_partial_match_for_multiple_warnings_messages():
    # [FUNCTION DOC]
    # Issue 56555. With the must-find-all option disabled, a matching future warning plus a non-matching user warning passes.
    # https://github.com/pandas-dev/pandas/issues/56555
    category = (FutureWarning, UserWarning)
    with tm.assert_produces_warning(
        category, match=r"^Match this", must_find_all_warnings=False
    ):
        warnings.warn("Match this", FutureWarning)  # pdlint: ignore[warning_class]
        warnings.warn("Not this", UserWarning)


def test_right_category_wrong_match_raises(pair_different_warnings):
    # [FUNCTION DOC]
    # Parametrized over pairs. A warning of the target category that does not match and a matching warning of another category raises an assertion error saying no warning matching the pattern was seen.
    target_category, other_category = pair_different_warnings
    with pytest.raises(AssertionError, match="Did not see warning.*matching"):
        with tm.assert_produces_warning(target_category, match=r"^Match this"):
            warnings.warn("Do not match it", target_category)
            warnings.warn("Match this", other_category)


@pytest.mark.parametrize("false_or_none", [False, None])
class TestFalseOrNoneExpectedWarning:
    # [CLASS DOC]
    # Tests of expecting no warnings; parametrized at class level over false and none.
    def test_raise_on_warning(self, false_or_none):
        # [FUNCTION DOC]
        # The helper that emits warnings, run inside the manager expecting none, raises an assertion error about unexpected warnings.
        msg = r"Caused unexpected warning\(s\)"
        with pytest.raises(AssertionError, match=msg):
            with tm.assert_produces_warning(false_or_none):
                f()

    def test_no_raise_without_warning(self, false_or_none):
        # [FUNCTION DOC]
        # An empty block with no expected warnings passes.
        with tm.assert_produces_warning(false_or_none):
            pass

    def test_no_raise_with_false_raise_on_extra(self, false_or_none):
        # [FUNCTION DOC]
        # With the extra-warning check disabled, the warning helper passes even though warnings are emitted.
        with tm.assert_produces_warning(false_or_none, raise_on_extra_warnings=False):
            f()


def test_raises_during_exception():
    # [FUNCTION DOC]
    # Three cases where the block raises an exception: expecting a user warning but the block raises a value error immediately gives an assertion error that the class was not seen; emitting only a future warning and then raising gives the same; expecting none but emitting a future warning then raising gives an assertion error about unexpected warnings.
    msg = "Did not see expected warning of class 'UserWarning'"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            raise ValueError

    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            warnings.warn(
                "FutureWarning", FutureWarning
            )  # pdlint: ignore[warning_class]
            raise IndexError

    msg = "Caused unexpected warning"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(None):
            warnings.warn(
                "FutureWarning", FutureWarning
            )  # pdlint: ignore[warning_class]
            raise SystemError


def test_passes_during_exception():
    # [FUNCTION DOC]
    # Expecting no warnings while the block raises a syntax error lets that error propagate; expecting a matching future warning that was emitted before a value error lets the value error propagate.
    with pytest.raises(SyntaxError, match="Error"):
        with tm.assert_produces_warning(None):
            raise SyntaxError("Error")

    with pytest.raises(ValueError, match="Error"):
        with tm.assert_produces_warning(FutureWarning, match="FutureWarning"):
            warnings.warn(
                "FutureWarning", FutureWarning
            )  # pdlint: ignore[warning_class]
            raise ValueError("Error")
