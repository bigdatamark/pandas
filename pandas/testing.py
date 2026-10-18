"""
Public testing utility functions.
"""

# [MODULE DOC]
# Public re-export module for pandas testing utilities.
# Re-exports four assertion helpers from the private testing module: extension array equality,
# frame equality, index equality and series equality. The public export list names exactly these four.
# Defines no functions or classes of its own and has no import-time side effects beyond the import.

from pandas._testing import (
    assert_extension_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

__all__ = [
    "assert_extension_array_equal",
    "assert_frame_equal",
    "assert_index_equal",
    "assert_series_equal",
]
