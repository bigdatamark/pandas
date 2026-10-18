# [MODULE DOC]
# Tests for the context manager that rewrites the category and/or message of a warning raised within its block when the original warning matches a target category and message substring.

import warnings

import pytest

from pandas.util._exceptions import rewrite_warning

import pandas._testing as tm


@pytest.mark.parametrize(
    "target_category, target_message, hit",
    [
        (FutureWarning, "Target message", True),
        (FutureWarning, "Target", True),
        (FutureWarning, "get mess", True),
        (FutureWarning, "Missed message", False),
        (DeprecationWarning, "Target message", False),
    ],
)
@pytest.mark.parametrize(
    "new_category",
    [
        None,
        DeprecationWarning,
    ],
)
def test_rewrite_warning(target_category, target_message, hit, new_category):
    # [FUNCTION DOC]
    # Parametrized over five (target category, target message, hit) cases (matching category and message or substring, missed message, different category) and over the new category being none or deprecation. Inside the rewrite context a future warning with the message "Target message" is emitted. If the target matched, the expected warning has the new category (or the original if none given) and the message "Rewritten message"; otherwise the original future warning and message pass through unchanged. Verified with the warning assertion helper.
    new_message = "Rewritten message"
    if hit:
        expected_category = new_category if new_category else target_category
        expected_message = new_message
    else:
        expected_category = FutureWarning
        expected_message = "Target message"
    with tm.assert_produces_warning(expected_category, match=expected_message):
        with rewrite_warning(
            target_message, target_category, new_message, new_category
        ):
            warnings.warn(
                message="Target message",
                category=FutureWarning,  # pdlint: ignore[warning_class]
            )
